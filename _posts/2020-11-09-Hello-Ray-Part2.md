---
layout: post
title: 'Hello Ray!  Part2: Build A Simple RL Demo'
categories: '强化学习'
tags:
  - [强化学习, ray]
---

In this part, we will use `python` to build a simple reinforcement learning demo to solve the games in `Open AI Gym` such as `CartPole`, `breakout`  and so on.

I absolutely believe that you must have a foundation in reinforcement learning if you want to practice `ray`. So in this series, I will not analyze details in RL or in those codes of demo. In this part, I will show you a simple RL demo to solve games in `Open AI Gym` and in next part, I will manage to parallelize this RL Demo with the help of `ray`.

## A Simple Reinforcement Learning Demo

### Model

Build a simple DQN model.

````python
"""
   models.py 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Dense neural network class."""
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 24)
        self.fc2 = nn.Linear(24, 48)
        self.out = nn.Linear(48, num_actions)

    def forward(self, states):
        """Forward pass."""
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.out(x)
````



### Common

Build some helper functions such as ReplayBuffer.

````python
"""
   common.py 
"""
import torch
import numpy as np
from collections import deque

class DataBuffer():
    def __init__(self, size , device):
        self.buffer = deque(maxlen=size)
        self.device = device 
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state,copy = False))
            actions.append(np.array(action, copy = False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy = False))
            dones.append(done)
            
        states = torch.as_tensor(np.array(states),device=self.device)
        actions = torch.as_tensor(np.array(actions),device=self.device)
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32),device=self.device)
        next_states = torch.as_tensor(np.array(next_states), device = self.device)
        dones = torch.as_tensor(np.array(dones, dtype=np.float32), device = self.device)
        return states, actions, rewards, next_states, dones
````



### Main

Train and test the model.

````python
"""
    main.py
"""
import model
import common
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discount = 0.99
loss_fn = nn.MSELoss()  
episodes = 1000
epsilon_max = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
batch_size = 64
max_len = 100000
lr = 0.01
gamma=0.01

def init_gym():
    env = gym.make("CartPole-v1")
    num_features = env.observation_space.shape[0]
    num_acitons = env.action_space.n

    return env, num_features, num_acitons


def select_epsilon_greedy_action(state, epsilon, env, main_nn):
    result = np.random.uniform()
    if result < epsilon:
        return env.action_space.sample()
    else:
        qs = main_nn(state).cpu().data.numpy()
        return np.argmax(qs)

def train_step(states, actions, rewards, next_states, dones, main_nn, target_nn, num_actions,optimizer):
    max_next_qs = target_nn(next_states).max(-1).values

    target = rewards + (1.0 - dones) * discount * max_next_qs

    qs = main_nn(states)

    action_masks = F.one_hot(actions, num_actions)
    masked_qs = action_masks.mul(qs).sum(-1)

    loss = loss_fn(masked_qs, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    epsilon = epsilon_max
    env, num_features, num_actions = init_gym()

    buffer = common.DataBuffer(max_len, device)

    main_nn = model.DQN(num_features, num_actions).to(device)
    target_nn = model.DQN(num_features, num_actions).to(device)
    
    optimizer = torch.optim.Adam(main_nn.parameters(), lr=lr)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.01, last_epoch=-1)

    last_100_ep_rewards = []

    cur_frame = 0
    for episode in range(episodes):
        state = env.reset().astype(np.float32)
        ep_reward = 0

        done = False
        while not done:
            state_in = torch.from_numpy(state)

            action = select_epsilon_greedy_action(state_in.to(device), epsilon, env, main_nn)
            
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)
            ep_reward = ep_reward + reward
            
            buffer.add(state, action, reward, next_state, done)
            state = next_state

            cur_frame += 1
            if cur_frame % 2000 == 0:
                target_nn.load_state_dict(main_nn.state_dict())

            if len(buffer) > batch_size and cur_frame > 10000 and cur_frame % 4 == 0:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                loss = train_step(states, actions, rewards, next_states, dones, main_nn, target_nn, num_actions, optimizer)

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        
        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)
        
        if episode % 50 == 0:
            print(f'Episode {episode}/{episodes}. Epsilon: {epsilon:.3f}.')
            print(f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.2f}')
        
    env.close()


if __name__ == "__main__":
    main()


````



## References

1. https://gym.openai.com/ 





## Further Reading

- [Hello Ray!  **Part1:** Ray Core Walkthrough](https://ysyisyourbrother.github.io/Hello-Ray-Part1/)

- [Hello Ray!  **Part2:** Build A Simple RL Demo](https://ysyisyourbrother.github.io/Hello-Ray-Part2/)        

- [Hello Ray!  **Part3:** Parallelize your RL model with ray](https://ysyisyourbrother.github.io/Hello-Ray-Part3/)      