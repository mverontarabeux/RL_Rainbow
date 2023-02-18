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

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Network, self).__init__()
        self.feature = NoisyLinear(state_dim, hidden_dim)  # input layer
        self.NL1 = NoisyLinear(hidden_dim, hidden_dim)  # hidden layer
        self.NL2 = NoisyLinear(hidden_dim, action_dim)  # output layer
        self.model = nn.Sequential(
            self.feature,
            nn.ReLU(),
            self.NL1,
            nn.ReLU(),
            self.NL2
        )

    def forward(self, x):
        # activation function
        return self.model(x)

    def reset_noise(self):
        # activation function
        self.NL1.reset_noise()
        self.NL2.reset_noise()


class noisyDQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.algo = cfg.algo_name

        self.action_dim = action_dim
        self.device = cfg.device  # cpu or gpu
        self.gamma = cfg.gamma  # discount factor
        self.frame_idx = 0  # attenuation
        self.batch_size = cfg.batch_size
        self.policy_net = Network(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = Network(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # copy parameters to target net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # optimizer
        self.memory = ReplayBuffer(cfg.memory_capacity)  # experience replay

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
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # transfer to tensor
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # calculate the expected q-value, for final state, done_batch[0]=1 and the corresponding
        # expected_q_value equals to reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        # update the network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # avoid gradient explosion by using clip
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Reset the noise 
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

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
    agent = noisyDQN(cfg.state_space_dim, cfg.action_space_dim, cfg)
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
        self.algo_name = 'noisyDQN' # algorithmic name
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