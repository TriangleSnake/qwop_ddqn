# Training an AI Agent to Play QWOP using DDQN

## Abstract

This project aims to train an artificial intelligence
agent capable to play the QWOP game using the
Double Deep Q-Network (DDQN) reinforcement
learning algorithm. QWOP is a physics-based game
known for its intricate control mechanics, making it a
challenging testbed for exploring reinforcement learning in complex and dynamic environments.
The proposed system framework involves interfacing
with the QWOP game simulator using the OpenAI
Gym, implementing the entire project in Python, and
utilizing PyTorch as the primary deep learning framework for constructing and training the DQN model.
We anticipate that through training, the AI agent will
exhibit progressively improving gameplay in QWOP

## Introduction
QWOP, a game that simulates real running, poses a
significant challenge for not just artificial intelligence
but also for human player.
The goal of this project is to develop an AI capable of playing the QWOP game using the Double
Deep Q-Network (DDQN) reinforcement learning algorithm. QWOP is a challenging physics-based game
that requires precise control of a runner’s limbs to
achieve forward motion. Training an AI agent to excel in this game presents a significant challenge due
to its complex dynamics and the need for high-level
coordination.

1. Game Environment: Instead of using the OpenAI Gym environment, we use
a modified version of the game environment
(https://github.com/yatshunlee/qwop_RL) that
rewrites the original JavaScript code. These modifications enable Selenium to get game data through
JavaScript commands. This allows the AI agent to
interact with the game through game/env.py and
obtain various values.

2. Machine Learning Framework: The entire
project will be implemented using Python programming language and PyTorch library as the
primary deep learning framework for constructing
and training the Deep Q-Network (DQN)

4. Training Model: The AI training model in this
project relies on the Deep Q-Network (DQN) and
its enhancement algorithm, Double Deep Q-Network
(DDQN) as the primary algorithms. DQN and
DDQN will be employed to learn optimal policies
for controlling the QWOP runner, with the aim of
achieving human-like running and obstacle navigation capabilities.

## Network Structure
Using a feedforward neural
network where the first layer inputs the x and y coordinates, as well as the x and y velocities, of various
parts of the character’s body. The hidden layer employs a ReLU function with 64 neurons. The final
fully connected layer represents the 11 possible combinations of key presses (q, w, o, p, qw, qo, qp, wo,
wp, op, and none).
Pre Processing: Execute JavaScript commands
through the environment set up by Selenium to obtain various of the character’s features.

## Result
### [DEMO video](https://youtu.be/AuSOg0PegPQ)
![image](https://github.com/TriangleSnake/qwop_ddqn/assets/46417323/8bc6ab82-564a-41fb-b24a-acf491036e80)




