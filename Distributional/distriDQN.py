import os
from typing import Dict, List, Tuple

import math
import random
import gym

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TradingSystem_v0:
    def __init__(self, returns_data, k_value, mode):
        self.mode = mode  # test or train
        self.index = 0
        self.data = returns_data
        self.tickers = list(returns_data.keys())
        self.current_stock = self.tickers[self.index]
        self.r_ts = self.data[self.current_stock]
        self.k = k_value
        self.total_steps = len(self.r_ts) - self.k
        self.current_step = 0
        self.initial_state = tuple(self.r_ts[:self.k])  # Use tuple because it's immutable
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False

    # write step function that returns obs(next state), reward, is_done
    def step(self, action):
        self.current_step += 1
        if self.current_step == self.total_steps:
            self.is_terminal = True
        self.reward = (action-1) * self.r_ts[self.current_step + self.k - 1]
        self.state = tuple(self.r_ts[self.current_step:(self.k + self.current_step)])
        return self.state, self.reward, self.is_terminal

    def reset(self):
        if self.mode == 'train':
            self.current_stock = random.choice(self.tickers)  # randomly pick a stock for every episode
        else:
            self.current_stock = self.tickers[self.index]
            self.index += 1
        self.r_ts = self.data[self.current_stock]
        self.total_steps = len(self.r_ts) - self.k
        self.current_step = 0
        self.initial_state = tuple(self.r_ts[:self.k])
        self.state = self.initial_state
        self.reward = 0.0
        self.is_terminal = False
        return self.state



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # capacity of buffer
        self.buffer = []  # replay buffer
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' replay buffer is a queue (LIFO)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, 
                support:torch.Tensor, atom_size=1, hidden_dim=128):
        super(Network, self).__init__()

        # Specific attribute for categorical (distributional) DQN
        self.support = support
        self.action_dim = action_dim
        self.atom_size = atom_size

        # Main network
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim * atom_size)  # output layer

        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        dist = self.dist(x)
        return torch.sum(dist * self.support, dim=2)

    def dist(self, x):
        # Compute the distribution of atoms
        q_atoms = self.model(x).view(-1, self.action_dim, self.atom_size)
        dist = F.softmax(q_atoms,dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist


class distriDQN:
    def __init__(self, state_dim, action_dim, cfg):

        self.action_dim = action_dim
        self.device = cfg.device  # cpu or gpu
        self.gamma = cfg.gamma  # discount factor
        self.frame_idx = 0  # attenuation
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = Network(state_dim, action_dim, hidden_dim=cfg.hidden_dim,
                                  support=cfg.support, atom_size=cfg.atom_size).to(self.device)
        self.target_net = Network(state_dim, action_dim, hidden_dim=cfg.hidden_dim,
                                  support=cfg.support, atom_size=cfg.atom_size).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # copy parameters to target net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # optimizer
        self.memory = ReplayBuffer(cfg.memory_capacity)  # experience replay

        self.v_max = cfg.v_max
        self.v_min = cfg.v_min
        self.atom_size = cfg.atom_size
        self.support = cfg.support

    def choose_action(self, state):
        # epsilon greedy policy
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()  # choose the action with maximum q-value
        else:
            action = random.randrange(self.action_dim) # Buy or sell
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # transfer to tensor
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        # Distributional DQN algo
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).argmax(1).detach()
            next_dist = self.target_net.dist(next_state_batch)
            next_dist = next_dist[range(self.batch_size), next_q_values]

            reward_batch = reward_batch.unsqueeze(1).expand_as(next_dist)
            done_batch = done_batch.unsqueeze(1).expand_as(next_dist)
            support = self.support.expand_as(next_dist)

            t_z = reward_batch + (1 - done_batch) * self.gamma * support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max) # Clip the t_z
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.policy_net.dist(state_batch)
        action_batch = action_batch.unsqueeze(1).expand(self.batch_size, 1, self.atom_size)
        dist = dist.gather(1, action_batch).squeeze()
        log_p = torch.log(dist)

        # COMPUTE THE LOSS NOW
        loss = -(proj_dist * log_p).sum(1).mean()
        
        # update the network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # avoid gradient explosion by using clip
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


def train(cfg, env, agent):
    ''' training
    '''
    print('Start Training!')
    print(f'Environment: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}')
    rewards = []  # record total rewards
    ma_rewards = []  # record moving average total rewards
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('Episode: {}/{}, Reward: {}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('Finish Training!')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}')
    ############# Test does not use e-greedy policy, so we set epsilon to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    ################################################################################
    stocks = env.tickers
    rewards = []  # record total rewards
    for i_ep in range(len(stocks)):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode: {i_ep + 1}/{len(stocks)}, Reward: {ep_reward:.1f}")
    print('Finish Testing!')
    return stocks, rewards


def env_agent_config(data, cfg, mode):
    ''' create environment and agent
    '''
    env = TradingSystem_v0(data, cfg.state_space_dim, mode)
    agent = distriDQN(cfg.state_space_dim, cfg.action_space_dim, cfg)
    if cfg.seed != 0:  # set random seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent


class Config:
    '''
    hyperparameters
    '''

    def __init__(self):
        ################################## env hyperparameters ###################################
        self.algo_name = 'distriDQN' # algorithmic name
        self.env_name = 'custom_trading_env' # environment name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # examine GPU
        self.seed = 11 # random seed
        self.train_eps = 200 # training episodes
        self.state_space_dim = 50 # state space size (K-value)
        self.action_space_dim = 3 # action space size (short: 0, neutral: 1, long: 2)
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 500  # capacity of experience replay
        self.batch_size = 64  # size of mini-batch SGD
        self.target_update = 4  # update frequency of target network
        self.hidden_dim = 128  # dimension of hidden layer
        # Specific parameters for distributional DQN below : 
        self.v_min = - 0.15
        self.v_max = 0.15
        self.atom_size = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)
        ################################################################################

        ################################# save path ##############################
        curr_path = os.path.dirname(__file__)
        curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'
        self.save = True  # whether to save the image
        ################################################################################


if __name__ == '__main__':

    import sys
    sys.path.insert(1, "C:\\Users\\mvero\\Desktop\\Cours\\M3\\Cours M3\\RL\\Projet\\RL_Rainbow\\")

    from Data.getdata import get_CAC40_tickers, get_tickers_data

    # Get the tickers on which we will train
    all_tickers = get_CAC40_tickers()
    nb_tickers = len(all_tickers)
    train_tickers = all_tickers[:10]
    test_tickers = all_tickers[-5:]

    # Set the dates
    start_date = '2015-01-02'
    end_date = '2023-02-15'

    # Get all returns 
    train_returns = get_tickers_data(tickers=train_tickers, 
                                    start_date=start_date,
                                    end_date=end_date,
                                    returns_only=True)
    print(f"Train returns shape = {train_returns.shape}")
    test_returns = get_tickers_data(tickers=test_tickers, 
                                    start_date=start_date,
                                    end_date=end_date,
                                    returns_only=True)
    print(f"Test returns shape = {test_returns.shape}")

    ################ parametrisation ################
    cfg = Config()

    ####### Create an environment and an agent #######
    env, agent = env_agent_config(train_returns, cfg, 'train')

    ################ Training ################
    rewards, ma_rewards = train(cfg, env, agent)
    os.makedirs(cfg.result_path)  # create output folders
    os.makedirs(cfg.model_path)
    agent.save(path=cfg.model_path)  # save model
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))   # plot the training result
    ax.plot(list(range(1, cfg.train_eps+1)), rewards, color='blue', label='rewards')
    ax.plot(list(range(1, cfg.train_eps+1)), ma_rewards, color='green', label='ma_rewards')
    ax.legend()
    ax.set_xlabel('Episode')
    plt.savefig(cfg.result_path+'train.jpg')

    ################ Testing ################
    env, agent = env_agent_config(test_returns, cfg, 'test')
    agent.load(path=cfg.model_path)  # load model
    stocks, rewards = test(cfg, env, agent)
    buy_and_hold_rewards = [sum(test_returns[stock]) for stock in stocks]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))  # plot the test result
    width = 0.3
    x = np.arange(len(stocks))
    ax.bar(x, rewards, width=width, color='salmon', label=cfg.algo_name)
    ax.bar(x+width, buy_and_hold_rewards, width=width, color='orchid', label='Buy and Hold')
    ax.set_xticks(x+width/2)
    ax.set_xticklabels(stocks, fontsize=12)
    ax.legend()
    plt.savefig(cfg.result_path+'test.jpg')