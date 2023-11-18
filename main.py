import torch
import threading
import time
import game.env as ENV
import host_game
import numpy as np
from collections import deque



INIT_EPSILON = 1.0
FINAL_EPSILON = 0.05
DURATION = 0.1
REPLAY_MEMORY_SIZE = 10000

STATE_SIZE = 41
ACTION_SIZE = 16 #qwop四個按鍵的組合

replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)


threading.Thread(target=host_game.launch_game).start() #host game server

env = ENV.QWOPEnv()

L1 = env.observation_space.shape[0]
L2 =150
L3 =100
L4 = env.action_space.n

#搞定nn
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


def get_reward(ret):
    return ret['torso']['position_x']

def get_state(state_dict):
    preprocessed_state = []

    # Extract head features
    head_features = ['angle', 'linear_velocity_x', 'linear_velocity_y', 'position_x', 'position_y']
    preprocessed_state.extend([state_dict['head'][feature] for feature in head_features])

    # Extract joint angles (you may choose to include other joint properties if needed)
    joint_features = ['leftAnkle', 'leftElbow', 'leftHip', 'leftKnee', 'leftShoulder',
                      'neck', 'rightAnkle', 'rightElbow', 'rightHip', 'rightKnee', 'rightShoulder']
    preprocessed_state.extend([state_dict['joints'][joint] for joint in joint_features])

    # Extract other body parts features (optional, based on relevancy)
    # Here, only angles are considered for simplicity
    body_parts = ['leftArm', 'leftCalf', 'leftFoot', 'leftForearm', 'leftThigh',
                  'rightArm', 'rightCalf', 'rightFoot', 'rightForearm', 'rightThigh', 'torso']
    for part in body_parts:
        preprocessed_state.append(state_dict[part]['angle'])

    return preprocessed_state


def train():
    while 1:
        ret = env.step(env.action_space.sample())
        print(ret[0])
        state = get_state(ret[0])
        reward = get_reward(ret[0])
        if env.gameover:
            env.reset()
        time.sleep(DURATION)


if __name__ == '__main__':
    train()

