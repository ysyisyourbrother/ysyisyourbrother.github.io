---
layout: post
title: 'Hello Ray!  Part3: Parallelize your RL model with ray'
tags:
  - [ray, RL]
---

In this part, we will use ray to design a simple `distributed reinforcement learning framework`, and put the code in the previous section into it.

## Design a Distributed RL Model

As shown in last part, traditional reinforcement learning model such as DQN is an iterative process. We can just set that only after agents have interacted with the environment serveral times will a reinforcement learning training be performed. 

<img src="https://ysyisyourbrother.github.io/images/posts_img/HelloRay/2.png" alt="image-20201101135934182" style="zoom: 50%;" />

However, if we want to parallelize this simple model, for example, we can have serveral independent agents who share same DQN parameters interact with environment at the same time or serveral training process which can move on parallel with agents, using the datas produced by interaction. And that is why ray stands out.

Parallel and distributed computing are a staple of modern applications. We need to leverage multiple cores or multiple machines to speed up applications or to run them at a large scale. **Ray provides a simple, universal API for building distributed applications.** 

With the help of ray, we can ignore communication(grpc) and storage(redis) and easily build a distributed RL model with sight modification. As shown below:

<img src="https://ysyisyourbrother.github.io/images/posts_img/HelloRay/3.jpeg" alt="image-20201101135934182" style="zoom:50%;" />

## Combining RL with the framework

In next serveral sections, I will show complete codes of the demo after combining the simple distributed RL model.

### Model

````python
"""
   models.py 
"""
class DQN(nn.Module):
    """
    Dense neural network class.

    state.dict() to get network parameters 
    load_state_dict() to load network parameters

    """

    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(num_inputs, 24)
        self.fc2 = nn.Linear(24, 48)
        self.out = nn.Linear(48, num_actions)

    def forward(self, states):
        """Forward pass."""
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.out(x)
````



### DataBuffer

````python
"""
	common.py
"""
class DataBuffer():
    def __init__(self, size, device):
        self.buffer = deque(maxlen=size)
        self.device = device
    
    def get_len(self):
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



### ParameterServer

PS is used to maintain parameters for main network. Agents refresh their policy network parameters after serveral actions, and trainers back up parameters to PS after each round of training.

````python
"""
	common.py
"""
@ray.remote
class ParameterServer(object):
    def __init__(self, parameters):
        self.parameters = parameters.copy()
    
    def get_weights(self):
        return self.parameters

    def set_weights(self, parameters):
        self.parameters = parameters.copy()

    def save(self):
        torch.save(self.parameters, "DQN.pt")
````



### WorkerAgents

We can start serveral workers to interact with environment in `main.py`, which can generate data in parallel.

````python
"""
	common.py
"""
@ray.remote
def worker_agents(PS_main, buffer, **kwargs):
    def select_epsilon_greedy_action(state, epsilon, env, main_nn):
        result = np.random.uniform()
        if result < epsilon:
            return env.action_space.sample()
        else:
            qs = main_nn(state).cpu().data.numpy()
            return np.argmax(qs)
    
    env = kwargs['env']
    device = kwargs['device']
    epsilon = kwargs['epsilon']

    main_nn = model.DQN(kwargs['num_features'],kwargs['num_actions']).to(device)

    last_100_ep_rewards = []

    for episode in range(kwargs['episodes']):
        # initiate environment
        state = env.reset().astype(np.float32)
        done = False

        ep_reward = 0

        # update main network
        main_nn.load_state_dict(ray.get(PS_main.get_weights.remote()))

        while not done:
            state_in = torch.from_numpy(state)
            # greedy algo.
            action = select_epsilon_greedy_action(state_in.to(device), epsilon , env, main_nn)

            # take action
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)
            ep_reward = ep_reward + reward
        
            # add buffer
            buffer.add.remote(state, action, reward, next_state, done)
            state = next_state

        # mutate e-greedy
        epsilon = max(kwargs['epsilon_min'], kwargs['epsilon_decay'] * epsilon)

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)
        

        episodes = kwargs['episodes']
        if episode % 50 == 0:
            print(f'Episode {episode}/{episodes}. Epsilon: {epsilon}')
            print(f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.2f}')

    env.close()
    return 0
````



### WorkerTrain

In training process, we need to pay attention to regularly backing up the parameters from policy network(main_nn) to PS and updating the target network.

````python
@ray.remote
def worker_train(PS_main, buffer, **kwargs):
    def train_step(states, actions, rewards, next_states, dones, main_nn, target_nn, optimizer, discount):
        max_next_qs = target_nn(next_states).max(-1).values

        target = rewards + (1.0 - dones) * discount * max_next_qs

        qs = main_nn(states)
        action_masks = F.one_hot(actions, main_nn.num_actions)
        masked_qs = action_masks.mul(qs).sum(-1)

        loss_fn = nn.MSELoss()  
        loss = loss_fn(masked_qs, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    device = kwargs['device']

    main_nn = model.DQN(kwargs['num_features'],kwargs['num_actions']).to(device)
    target_nn = model.DQN(kwargs['num_features'],kwargs['num_actions']).to(device)

    # optimizer
    optimizer = torch.optim.Adam(main_nn.parameters(), lr=kwargs['lr'])

    # train
    cur_step = 0 
    for ep in range(kwargs['episodes']):
        for bn in range(kwargs['batch_num']):
            flag = False
            if flag or ray.get(buffer.get_len.remote()) >= 10000:
                # avoid continuously .remote()
                flag = True
                cur_step += 1

                states, actions, rewards, next_states, dones = ray.get(buffer.sample.remote(kwargs['batch_size']))
                train_step(states, actions, rewards, next_states, dones, main_nn, target_nn, optimizer, kwargs['discount'])

                # back up main_nn to PS_main
                PS_main.set_weights.remote(main_nn.state_dict())

                # copy para. in main_nn to target_nn
                if cur_step % 1000 == 0:
                    target_nn.load_state_dict(main_nn.state_dict())
        if ep % 30 ==0:
            episodes = kwargs['episodes']
            print(f"training epoch: {ep}/{episodes}")
````



### main

In main function, we can customize the number of parallel agents and hyperparameters.

````python
import ray
import model
from common import * 

import time


def init_gym():
    env = gym.make("CartPole-v1")
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    return env, num_features, num_actions

def main():
    """
        main
    """
    ray.init()

    env, num_features, num_actions = init_gym()

    device = torch.device("cpu")

    num_features, num_actions = 4, 2

    # initiate
    net = model.DQN(num_features,num_actions).to(device)
    parameters = net.state_dict()

    PS_main = ParameterServer.remote(parameters)
    buffer = DataBuffer.remote(100000, device)
    
    res = []
    roll_out_num = 2
    for _ in range(roll_out_num):
        res.append(worker_agents.remote(PS_main,buffer,
                        device = device,
                        epsilon = 1,
                        episodes = 3000,
                        epsilon_min = 0.01,
                        epsilon_decay = 0.9995,
                        env = env,
                        num_features = num_features,
                        num_actions = num_actions,))
    # time.sleep(3)

    res.append(worker_train.remote(PS_main, buffer, 
                  device = device,
                  lr = 0.001,
                  episodes = 1000,
                  batch_num = 10,
                  batch_size = 64,
                  discount = 0.99,
                  env = env,
                  num_features = num_features,
                  num_actions = num_actions,))

    ray.get(res)

if __name__ == "__main__":
    main()
````



### References

1. https://docs.ray.io/en/master/serialization.html



### Further Reading

- [Hello Ray!  **Part1:** Ray Core Walkthrough](https://ysyisyourbrother.github.io/posts/2020/11/blog-post-1/)

- [Hello Ray!  **Part2:** Build A Simple RL Demo](https://ysyisyourbrother.github.io/posts/2020/11/blog-post-2/)        

- [Hello Ray!  **Part3:** Parallelize your RL model with ray](https://ysyisyourbrother.github.io/posts/2020/11/blog-post-3/)        