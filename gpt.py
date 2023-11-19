import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import game.env as ENV
# 定义Q网络的类
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(state_size, hidden_size)
        # 定义激活函数
        self.relu = nn.ReLU()
        # 定义第二个全连接层（输出层）
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        # 前向传播过程，返回给定状态的动作价值
        x = self.relu(self.fc1(state))
        return self.fc2(x)

# 初始化环境和Q网络
env = ENV.QWOPEnv()  # 你的环境

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
q_network = QNetwork(state_size, action_size)
target_q_network = QNetwork(state_size, action_size)
# 将Q网络的权重复制到目标Q网络
target_q_network.load_state_dict(q_network.state_dict())

# 设置优化器和损失函数
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
# 初始化经验回放存储
replay_memory = deque(maxlen=10000)
# 设置折扣因子和探索率
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
update_target_every = 5  # 多少回合更新一次目标网络
batch_size = 64
num_episodes = 1000
max_steps_per_episode = 1000
# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    print(state)
    state = torch.tensor(state, dtype=torch.float32)
    for t in range(max_steps_per_episode):
        # 根据epsilon贪婪策略选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()  # 随机动作
        else:
            q_values = q_network(state.unsqueeze(0)).detach()
            action = q_values.max(1)[1].item()  # 最大Q值的动作
        print(action)
        next_state, reward, done, false, _ = env.step(action)
        
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # 将转换存储到经验回放存储中
        replay_memory.append((state, action, reward, next_state, done))
        
        # 经验回放
        if len(replay_memory) > batch_size:
            # 从经验回放存储中随机采样一个小批量转换
            minibatch = random.sample(replay_memory, batch_size)
            # 解包转换
            states, actions, rewards, next_states, dones = zip(*minibatch)
            
            # 转换为适合网络输入的张量
            states = torch.stack(states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            # 使用目标Q网络计算下一个状态的最大预期Q值
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_q_network(next_states).max(1)[0]
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            # 计算损失并执行梯度下降
            loss = loss_fn(current_q_values.float(), expected_q_values.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        
        # 如果回合结束则跳出循环
        if done:
            break
    
    # 更新epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # 定期更新目标网络的权重
    if episode % update_target_every == 0:
        target_q_network.load_state_dict(q_network.state_dict())

    # 输出训练信息
    print(f"Episode {episode} finished after {t+1} timesteps")

