import os
from typing import Dict, List, Tuple

import datetime as dt
import torch

dict_algo_folder = {
    "DQN":"DQN",
    "DDQN":"DDQN",
    "duelingDQN":"Dueling",
    "distriDQN":"Distributional",
    "multistepDQN":"MultiStep",
    "noisyDQN":"Noisy",
    "prioDQN":"Prioritized",
    "rainbow":"",
}

class ConfigBase:
    '''
    hyperparameters
    '''

    def __init__(self, algo="DQN"):
        ################################## env hyperparameters ###################################
        self.algo_name = algo # algorithmic name
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
        ################################################################################

        ################################# save path ##############################
        curr_path = os.path.dirname(__file__)
        curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        middle_path = "\\" + dict_algo_folder[algo] if algo !="Rainbow" else ""
        self.result_path = curr_path + middle_path +  "\\outputs\\" 
        self.model_path = curr_path + middle_path +  "\\outputs\\" 
        self.save = True  # whether to save the image
        ################################################################################

def test_with_returns(cfg, env, agent):
    print(f'Start Testing for {cfg.algo_name} !')
    print(f'Environment: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}')
    
    stocks = env.tickers
    rewards = []  # record total rewards
    rewards_series = []  # record rewards series
    for i_ep in range(len(stocks)):
        ep_reward = 0
        state = env.reset()
        new_serie = []
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            new_serie.append(reward)
            if done:
                break
        rewards.append(ep_reward)
        rewards_series.append(new_serie)
        print(f"Episode: {i_ep + 1}/{len(stocks)}, Reward: {ep_reward:.1f}")
    print('Finish Testing!\n\n')

    return stocks, rewards, rewards_series


if __name__ == '__main__':
    # TESTING ALL SAVED ALGORITHM AND COMPARE THE RETURNS

    import sys
    sys.path.insert(1, "C:\\Users\\mvero\\Desktop\\Cours\\M3\\Cours M3\\RL\\Projet\\RL_Rainbow\\")

    #####################################################
    # Import all the modules
    #####################################################
    import pandas as pd
    import numpy as np 
    import matplotlib.pyplot as plt
    from Data.getdata import get_CAC40_tickers, get_tickers_data
    from DQN import DQN
    from DDQN import DDQN
    from Dueling import duelingDQN
    from Distributional import distriDQN
    from MultiStep import multistepDQN
    from Noisy import noisyDQN
    from Prioritized import prioDQN
    import rainbow

    random_seed = 42
    torch.manual_seed(random_seed)

    #####################################################
    # Get the testing data 
    #####################################################

    # Get the last 5 tickers on which we will test the trading bot
    all_tickers = get_CAC40_tickers()
    test_tickers = all_tickers[-5:]

    # Set the dates
    test_start_date = '2022-01-02'
    test_end_date = '2023-02-15'

    # Get all returns 
    test_returns = get_tickers_data(tickers=test_tickers, 
                                    start_date=test_start_date,
                                    end_date=test_end_date,
                                    returns_only=True)
    print(f"Test returns shape = {test_returns.shape}")

    #####################################################
    # TESTING ALL SAVED ALGORITHM AND COMPARE THE RETURNS
    #####################################################

    dict_series_by_algo = {}
    ########## DQN ##########   
    config = ConfigBase("DQN")

    env, agent = DQN.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["DQN"] = df.cumsum()
    
    ########## DDQN ##########
    config = ConfigBase("DDQN")

    env, agent = DDQN.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["DDQN"] = df.cumsum()

    ########## Distributional DQN ##########
    config = ConfigBase("distriDQN")
    config.v_min = - 0.15
    config.v_max = 0.15
    config.atom_size = 51
    config.support = torch.linspace(config.v_min, config.v_max, config.atom_size).to(config.device)

    env, agent = distriDQN.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["distriDQN"] = df.cumsum()

    ########## Dueling DQN ##########
    config = ConfigBase("duelingDQN")

    env, agent = duelingDQN.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["duelingDQN"] = df.cumsum()

    ########## MultiStep DQN ##########
    config = ConfigBase("multistepDQN")
    config.n_step = 10

    env, agent = multistepDQN.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["multistepDQN"] = df.cumsum()

    ########## Noisy DQN ##########
    config = ConfigBase("noisyDQN")
    # Nothing to add (epsilon removed in the class)

    env, agent = noisyDQN.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["noisyDQN"] = df.cumsum()


    ########## Prioritized DQN ##########
    config = ConfigBase("prioDQN")
    config.alpha = 0.2
    config.beta = 0.6
    config.prior_eps = 1e-6

    env, agent = prioDQN.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["prioDQN"] = df.cumsum()

    ########## Rainbow DQN ##########
    config = ConfigBase("rainbow")
    # Multistep Learning parameters
    config.n_step = 10
    # PER parameters
    config.alpha = 0.2
    config.beta = 0.6
    config.prior_eps = 1e-6
    # Categorical DQN parameters
    config.v_min = - 0.15
    config.v_max = 0.15 # Proxy of the daily returns MINMAX range
    config.atom_size = 51
    config.support = torch.linspace(config.v_min, config.v_max, config.atom_size).to(config.device)

    env, agent = rainbow.env_agent_config(test_returns, config, 'test')
    agent.load(path=config.model_path)  # load model
    stocks, rewards, rewards_series = test_with_returns(config, env, agent)
    df = pd.DataFrame({stocks[i]:rewards_series[i] for i in range(len(stocks))})
    dict_series_by_algo["rainbow"] = df.cumsum()

    #####################################################
    # Formatting the dictionnary of series by algorithms
    #####################################################

    full_df = pd.concat(dict_series_by_algo.values(), axis=1) 
    full_df.columns = [full_df.columns, list(np.repeat(list(dict_series_by_algo.keys()),len(test_tickers)))]

    full_df_index = full_df.swaplevel(axis=1)
    
    #####################################################
    # Define and save the plot
    #####################################################
    all_algo_ordered = ["DQN","DDQN","Distributional", "Dueling", "Multi-step","Noisy","Prioritized","Rainbow"]
    all_algo_method_ordered = ["DQN","DDQN","distriDQN", "duelingDQN", "multistepDQN","noisyDQN","prioDQN","rainbow"]

    for ticker in test_tickers:
        plt.figure(figsize=(9,4))
        plt.plot(full_df[ticker].iloc[:,:-1], linewidth=1)
        std_list = list(full_df[ticker].iloc[:,:-1].std())
        leg = [f"{all_algo_ordered[i]} _ std={round(std_list[i],3)}" for i in range(len(all_algo_ordered[:-1]))]

        # Add the rainbow, thicker
        plt.plot(full_df[ticker].iloc[:,-1], color='black', linewidth=3)
        std_list.append(full_df[ticker].iloc[:,-1].std())
        leg.append(f"{all_algo_ordered[-1]} _ std={round(std_list[-1],3)}")

        # Add the buy and hold
        buy_and_hold = test_returns[ticker].cumsum()[50:].reset_index(drop=True) # 50 because of the K in the algos
        std_list.append(buy_and_hold.std())
        plt.plot(buy_and_hold, color='red', linewidth=2)
        leg.append(f"Buy and Hold _ std={round(std_list[-1],3)}")

        plt.axhline(y=0, color='black', linestyle='-')
        plt.title(ticker)
        plt.plot(buy_and_hold, color='red', linewidth=2, label=f"Buy and Hold _ std={round(std_list[-1],3)}")
        plt.legend(leg, bbox_to_anchor=(1, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(ticker + ".jpeg")
        plt.clf()

    #####################################################
    # Get equally-weighted portfolio by algo
    #####################################################
    std_list = []
    # All but Rainbow
    plt.figure(figsize=(10,5))
    for i in range(len(all_algo_method_ordered[:-1])):
        algo_method = all_algo_method_ordered[i]
        algo = all_algo_ordered[i]
        equipondere_index = full_df_index[algo_method].mean(axis=1)
        std_list.append(equipondere_index.std())
        plt.plot(equipondere_index, linewidth=1, label=f"{algo} _ std={round(std_list[i],3)}")

    # Rainbow now
    equipondere_index = full_df_index[all_algo_method_ordered[-1]].mean(axis=1)
    std_list.append(equipondere_index.std())
    plt.plot(equipondere_index, color='black', linewidth=3, label=f"{all_algo_ordered[-1]} _ std={round(std_list[-1],3)}")

    # And finally buy and hold
    buy_and_hold = test_returns.mean(axis=1).cumsum()[50:].reset_index(drop=True) # 50 because of the K in the algos
    std_list.append(buy_and_hold.std())
    plt.plot(buy_and_hold, color='red', linewidth=2, label=f"Buy and Hold _ std={round(std_list[-1],3)}")

    plt.axhline(y=0, color='black', linestyle='-')
    plt.title("Equally-weighted portfolio by algorithm")
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig("equallyweighted.jpeg")
    plt.clf()
