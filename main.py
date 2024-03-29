import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import game.env as ENV
import pickle

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        return self.fc2(x)

def save_model():
    print("Saving model...")
    torch.save(q_network.state_dict(),'q_network.pth')
    torch.save(target_q_network.state_dict(),'target_q_network.pth')
    with open("variables.pkl","wb") as f:
        variables = {
            "epsilon":epsilon,
            "episode":episode,
            "replay_memory":replay_memory
        }
        pickle.dump(variables,f)
    print("Model saved")

def load_model():
    global q_network,target_q_network,gamma,epsilon,min_epsilon,epsilon_decay,update_target_every,batch_size,num_episodes,max_steps_per_episode,episode,replay_memory
    try:
        q_network.load_state_dict(torch.load("q_network.pth"))    
        target_q_network.load_state_dict(torch.load("target_q_network.pth"))
        with open("variables.pkl","rb") as f:
            variables = pickle.load(f)
        epsilon = variables["epsilon"]
        episode = variables["episode"]
        replay_memory = variables['replay_memory']
        print("Loaded model!")
    except Exception as e:
        print("Failed to load model...",e)
        q_network = QNetwork(state_size,action_size)
        target_q_network = QNetwork(state_size,action_size)

env = ENV.QWOPEnv()  

state_size = 71
action_size = env.action_space.n

q_network = QNetwork(state_size, action_size)
target_q_network = QNetwork(state_size, action_size)

# copy q network to traget q network
target_q_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# hyperpamater
replay_memory = deque(maxlen=10000)
gamma = 0.95
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.99
update_target_every = 5  # 多少回合更新一次目标网络
batch_size = 64
num_episodes = 10000
max_steps_per_episode = 200 
episode = 0
max_episode = 10000

#中斷後嘗試load_model(如果沒有則創一個新的)
load_model()

def get_state(state_dict):
    preprocessed_state = []
    
    
    features = ['angle', 'linear_velocity_x', 'linear_velocity_y', 'position_y','position_x']

    joint_features = ['leftAnkle', 'leftElbow', 'leftHip', 'leftKnee', 'leftShoulder',
                      'neck', 'rightAnkle', 'rightElbow', 'rightHip', 'rightKnee', 'rightShoulder']
    preprocessed_state.extend([state_dict['joints'][joint] for joint in joint_features])

    
    body_parts = ['head','leftArm', 'leftCalf', 'leftFoot', 'leftForearm', 'leftThigh',
                  'rightArm', 'rightCalf', 'rightFoot', 'rightForearm', 'rightThigh', 'torso']
    for part in body_parts:
        preprocessed_state.extend([state_dict[part][feature] for feature in features])
    return preprocessed_state

def get_reward(ret):
    
    return ret['reward'] 




prev_pos_x = 0
while 1:
    state = env.reset()
    state = get_state(state)
    state = torch.tensor(state, dtype=torch.float32)
    for t in range(max_steps_per_episode):

        if random.random() < epsilon:
            print("\rrandom",end="")
            action = env.action_space.sample()
        else:
            print("\rchoose",end="")
            q_values = q_network(state.unsqueeze(0)).detach()
            action = q_values.max(1)[1].item()  # 最大Q value的動作
        data = env.step(action)[0]
        next_state = get_state(data)
        reward = get_reward(data)
        done = env.gameover

        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        
        replay_memory.append((state, action, reward, next_state, done))
        

        # 經驗回放
        if len(replay_memory) > batch_size:
            minibatch = random.sample(replay_memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            
            #預處理成tensor
            states = torch.stack(states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            # target_q_network取得最大預期Q值
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_q_network(next_states).max(1)[0]
            #print([type(i) for i in [reward,gamma,next_q_values,dones]])
            expected_q_values = rewards + float(gamma) * next_q_values * (1 - dones)
            
            #計算損失&梯度下降
            loss = loss_fn(current_q_values.float(), expected_q_values.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        
        if done:
            break
    
    # 更新epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # 每 update_target_every 個回合將target network更新為當前q_network
    if episode % update_target_every == 0:
        target_q_network.load_state_dict(q_network.state_dict())
        save_model()
        
    
    print(f"Episode {episode} finished after {t+1} timesteps,epsilon:{epsilon},dist:{data['score']}")
    with open('score.data','a') as f:
        f.write(str(data['score'])+'\n')
    if episode >= max_episode:
        break
    episode += 1
