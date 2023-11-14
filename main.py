import torch
import threading
import time
import game.env as ENV
import host_game
import numpy as np

INIT_EPSILON = 1.0
FINAL_EPSILON = 0.05
DURATION = 0.1



threading.Thread(target=host_game.launch_game).start() #host game server

env = ENV.QWOPEnv()

L1 = env.observation_space.shape[0]
L2 =150
L3 =100
L4 = env.action_space.n

model = torch.nn.Sequential(
    torch.nn .Linear(L1, L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2, L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3, L4)
)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print(f"L1:{L1}, L2:{L2}, L3:{L3}")
def get_reward(ret):
    return 0

def get_state(state):
    ret_state = np.array([])
    for i in state:
        if type(i) != dict:
            ret_state.append(i)
        else :
            for j in i.values():
                ret_state.append(j)
    

def train():
    while 1:
        ret = env.step(env.action_space.sample())
        print(ret[0])
        state = get_state(ret[0])
        reward = get_reward(state)
        if env.gameover:
            env.reset()
        time.sleep(DURATION)


if __name__ == '__main__':
    train()

