import os
from typing import Dict, List, Tuple, Deque

import math
import random

from collections import deque
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DQN.DQN import TradingSystem_v0
from MultiStep.multistepDQN import MultiStepReplayBuffer
from Prioritized.prioDQN import PrioritizedReplayBuffer
from Noisy.noisyDQN import NoisyLinear
from Prioritized.segment_tree import MinSegmentTree, SumSegmentTree


class RainbowNetwork(nn.Module):
    """Combine NoisyNet + DuelingNet + Categorical DQN """
    def __init__(self, state_dim, action_dim, 
                support:torch.Tensor, atom_size=1, hidden_dim=128):
        super(RainbowNetwork, self).__init__()

        # Specific attribute for distributional DQN
        self.support = support
        self.action_dim = action_dim
        self.atom_size = atom_size

        # Feature layer for Dueling
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(),
        )

        # Advantage layer for Dueling, with noisy net and distri
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * self.atom_size),
        )

        # Value layer for Dueling, with noisy net and distri
        self.value = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, self.atom_size),
        )

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x):
        # Get feature/value from dueling
        feature = self.feature(x)
        value = self.value(feature)[:,None,:]
        advantage = self.advantage(feature).view(-1, self.action_dim, self.atom_size)
        # Compute the distribution of atoms
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms,dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist

    def reset_noise(self):
        self.advantage.reset_noise()
        self.value.reset_noise()


class Rainbow:
    def __init__(self, state_dim, action_dim, cfg):
        self.algo = cfg.algo_name
        self.action_dim = action_dim
        self.device = cfg.device  # cpu or gpu
        self.gamma = cfg.gamma  # discount factor
        self.frame_idx = 0  # attenuation

        self.batch_size = cfg.batch_size
        
        # RainbowNetworks 
        self.policy_net = RainbowNetwork(state_dim, action_dim, hidden_dim=cfg.hidden_dim,
                                  support=cfg.support, atom_size=cfg.atom_size).to(self.device)
        self.target_net = RainbowNetwork(state_dim, action_dim, hidden_dim=cfg.hidden_dim,
                                  support=cfg.support, atom_size=cfg.atom_size).to(self.device)

        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # copy parameters to target net
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # optimizer

    
        # Prioritized parameters
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.prior_eps = cfg.prior_eps
        self.memory = PrioritizedReplayBuffer(cfg.memory_capacity, 
                                              batch_size=self.batch_size,
                                              alpha=self.alpha)  # 1 step experience replay

        # Multi step parameters
        self.n_step = cfg.n_step
        self.n_step_memory = MultiStepReplayBuffer(cfg.memory_capacity, 
                                                   n_step=10,
                                                   gamma=self.gamma)  # n step experience replay
        
        # Distributional parameters
        self.v_max = cfg.v_max
        self.v_min = cfg.v_min
        self.atom_size = cfg.atom_size
        self.support = cfg.support


    def choose_action(self, state):
        self.frame_idx += 1
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()  # choose the action with maximum q-value
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # Sample first with prio
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = self.memory.sample(
            self.beta)

        # 1 step learning loss
        elementwise_loss = self.rainbow_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                                             self.gamma)

        # n step learning loss 
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.n_step_memory.sample_batch_from_idxs(
            indices, self.device)
        gamma = self.gamma ** self.n_step
        elementwise_n_loss = self.rainbow_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                                               gamma)
        elementwise_loss += elementwise_n_loss   

        # Prioritized weighting : 
        loss = torch.mean(elementwise_loss * weights)

        # update the network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # avoid gradient explosion by using clip
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update the priority scheme
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # Reset noise for noisy networks
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def rainbow_loss(self, state, action, reward, next_state, done,
                    gamma):
        # transfer to tensor
        state_batch = torch.tensor(state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done), device=self.device)

        # Distributional DQN algo
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        with torch.no_grad():
            # DDQN
            next_q_values = self.policy_net(next_state_batch).argmax(1).detach() # policy
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

        # COMPUTE THE LOSS ELEMENTWISE (removing the mean from distri)
        elementwise_loss  = -(proj_dist * log_p).sum(1)

        return elementwise_loss 


    def save(self, path):
        torch.save(self.target_net.state_dict(), path + self.algo + '_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + self.algo + '_checkpoint.pth'))
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

            # save transition in the n_step_memory
            n_state, n_action, n_reward, n_next_state, n_done = agent.n_step_memory.push(state, 
                                                                               action, 
                                                                               reward, 
                                                                               next_state, 
                                                                               done)  
            
            if not all(element is None for element in (n_state, n_action, n_reward, n_next_state, n_done)):
                # save the one step one in the memory
                agent.memory.push(n_state, n_action, n_reward, n_next_state, n_done)

            state = next_state
            agent.update()
            ep_reward += reward

            # Prio : increase beta
            fraction = min(i_ep / cfg.train_eps, 1.0)
            agent.beta = agent.beta + fraction * (1.0 - agent.beta) 

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
    agent = Rainbow(cfg.state_space_dim, cfg.action_space_dim, cfg)
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
        self.algo_name = 'Rainbow' # algorithmic name
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
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 500  # capacity of experience replay
        self.batch_size = 64  # size of mini-batch SGD
        self.target_update = 4  # update frequency of target network
        self.hidden_dim = 128  # dimension of hidden layer
        
        # N-step Learning
        self.n_step = 10

        # PER parameters
        self.alpha = 0.2
        self.beta = 0.6
        self.prior_eps = 1e-6

        # Categorical DQN parameters
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