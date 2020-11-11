---
layout: post
title: 'Hello Ray!  Part2: Build A Simple RL Demo'
date: 2020-11-09
permalink: /posts/HelloRay2/
tags:
  - [ray, RL]
---

In this part, we will use `python` to build a simple reinforcement learning demo to solve the games in `Open AI Gym` such as `CartPole`, `breakout`  and so on.

I absolutely believe that you must have a foundation in reinforcement learning if you want to practice `ray`. So in this series, I will not analyze details in RL or in those codes of demo. In this part, I will show you a simple RL demo to solve games in `Open AI Gym` and in next part, I will manage to parallelize this RL Demo with the help of `ray`.

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
    """
       缓存游戏过程样本数据
       通过随机采样的方式采样batch数据提供训练DQN
    """

    def __init__(self, size , device):
        self.buffer = deque(maxlen=size)
        self.device = device    # 判断是否可以使用cuda
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, num_samples):
        """
            采样数据并组装成tensor作为训练输入
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            # 将数据各个归类打包
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


# 判断设备是cpu还是gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bellman 公式中对next_state_qs的削减值
discount = 0.99
# 损失函数为平方误差
loss_fn = nn.MSELoss()  
# 总迭代轮次
episodes = 1000
# greedy算法随机概率 逐渐递减调整 
epsilon_max = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
# 训练的时候每批数据量
batch_size = 64
# buffer总的数据量
max_len = 100000
# 学习率
lr = 0.01
# 学习率衰减
gamma=0.01

def init_gym():
    """
        初始化gym游戏环境
    """
    env = gym.make("CartPole-v1")
    num_features = env.observation_space.shape[0]
    num_acitons = env.action_space.n

    return env, num_features, num_acitons


def select_epsilon_greedy_action(state, epsilon, env, main_nn):
    """
        贪婪算法模拟游戏进程
    """
    result = np.random.uniform()
    if result < epsilon:
        # 随机选择action来执行
        return env.action_space.sample()
    else:
        # 计算Q(s, a)选择最大的a来执行
        qs = main_nn(state).cpu().data.numpy()

        # 返回值最大位置的index，作为action
        return np.argmax(qs)

def select_greedy_action(state, main_nn):
    """
        返回qs最大的最优action
    """
    # 不能直接将cuda的数据转为numpy
    qs = main_nn(state).cpu().data.numpy()
    return np.argmax(qs)

def train_step(states, actions, rewards, next_states, dones, main_nn, target_nn, num_actions,optimizer):
    """
        每一次DQN训练（minibatch）需要计算的东西
        包括bellman函数、损失函数、反向传播
    """
    # 计算S(s', a')，求出下一状态s'中，对于所有a'可能的最大reward
    max_next_qs = target_nn(next_states).max(-1).values

    # 利用bellman公式，计算Q(s,a)
    target = rewards + (1.0 - dones) * discount * max_next_qs

    # 计算当前状态的qs
    qs = main_nn(states)
    # 将action组装为onehot矩阵
    action_masks = F.one_hot(actions, num_actions)
    # onehot取出当前状态
    masked_qs = action_masks.mul(qs).sum(-1)

    # 损失函数是让masked_qs尽量接近target
    # detach得到的这个Variable永远不需要计算其梯度
    loss = loss_fn(masked_qs, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    """
        主函数
    """
    epsilon = epsilon_max
    # 初始化gym
    env, num_features, num_actions = init_gym()

    # 初始化buffer
    buffer = common.DataBuffer(max_len, device)

    # 构建DQN
    main_nn = model.DQN(num_features, num_actions).to(device)
    target_nn = model.DQN(num_features, num_actions).to(device)
    
    # 定义损失函数
    optimizer = torch.optim.Adam(main_nn.parameters(), lr=lr)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.01, last_epoch=-1)

    last_100_ep_rewards = []
    # 一共采用了多少次action 
    cur_frame = 0
    # 一共训练episodes轮游戏
    for episode in range(episodes):
        # 初始化状态
        state = env.reset().astype(np.float32)
        # 每一轮游戏总收益
        ep_reward = 0

        # 游戏是否结束
        done = False
        while not done:
            state_in = torch.from_numpy(state)
            # 使用greedy算法选择下一步
            action = select_epsilon_greedy_action(state_in.to(device), epsilon, env, main_nn)

            # 在环境中执行action
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)
            ep_reward = ep_reward + reward
            
            # 将结果加入buffer中用来训练
            buffer.add(state, action, reward, next_state, done)
            state = next_state

            cur_frame += 1
            # 每采取5000次action，就将数据复制进target_nn中 
            if cur_frame % 2000 == 0:
                target_nn.load_state_dict(main_nn.state_dict())

            # 训练神经网络
            if len(buffer) > batch_size and cur_frame > 10000 and cur_frame % 4 == 0:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                loss = train_step(states, actions, rewards, next_states, dones, main_nn, target_nn, num_actions, optimizer)

        # 动态改变e-greedy算法的e值
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



### References

1. https://gym.openai.com/ 





### Further Reading

- [Hello Ray!  **Part1:** Ray Core Walkthrough](https://ysyisyourbrother.github.io/posts/2020/11/blog-post-1/)

- [Hello Ray!  **Part2:** Build A Simple RL Demo](https://ysyisyourbrother.github.io/posts/2020/11/blog-post-2/)        

- [Hello Ray!  **Part3:** Parallelize your RL model with ray](https://ysyisyourbrother.github.io/posts/2020/11/blog-post-3/)        