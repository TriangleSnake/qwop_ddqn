import torch
import torch.nn as nn
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

def get_state(state_dict):
    preprocessed_state = []
    # Extract head features
    features = ['angle', 'linear_velocity_x', 'linear_velocity_y', 'position_y','position_x']

    # Extract joint angles (you may choose to include other joint properties if needed)
    joint_features = ['leftAnkle', 'leftElbow', 'leftHip', 'leftKnee', 'leftShoulder',
                      'neck', 'rightAnkle', 'rightElbow', 'rightHip', 'rightKnee', 'rightShoulder']
    preprocessed_state.extend([state_dict['joints'][joint] for joint in joint_features])

    # Extract other body parts features (optional, based on relevancy)
    # Here, only angles are considered for simplicity
    body_parts = ['head','leftArm', 'leftCalf', 'leftFoot', 'leftForearm', 'leftThigh',
                  'rightArm', 'rightCalf', 'rightFoot', 'rightForearm', 'rightThigh', 'torso']
    for part in body_parts:
        preprocessed_state.extend([state_dict[part][feature] for feature in features])
    return preprocessed_state

env = ENV.QWOPEnv()

state_size = 71
action_size = env.action_space.n


q_network = QNetwork(state_size, action_size)
import sys
try:
    path = './trained_data/'+sys.argv[1]
except:
    path = '.'

q_network.load_state_dict(torch.load(path + "/q_network.pth"))
state = env.reset()
while 1:
    state = get_state(state)
    state = torch.tensor(state, dtype=torch.float32)
    q_values = q_network(state.unsqueeze(0)).detach()
    action = q_values.max(1)[1].item()
    state = env.step(action)[0]
    if env.gameover:
        state = env.reset()