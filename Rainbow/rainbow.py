from Prioritized.segment_tree import MinSegmentTree, SumSegmentTree

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

from DQN import TradingSystem_v0
from MultiStep.mu import ReplayBuffer
from Prioritized.prioDQN import PrioritizedReplayBuffer
from Noisy.noisyDQN import NoisyLinear

class ReplayBuffer:
    def __init__(self, capacity, n_step=1, gamma=0.99):
        self.capacity = capacity  # capacity of buffer
        self.buffer = []  # replay buffer
        self.position = 0

        # for multistep Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, state, action, reward, next_state, done):
        ''' replay buffer filling
        '''
        # Set the transition as tuple
        transition = (state, action, reward, next_state, done)
        # Add it everytime to the deque
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return (None, None, None, None, None)

        # Increase the dimension of the buffer if needed
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        # Get the multi step transition
        reward, next_state, done = self._get_n_step_info(
            self.gamma, self.n_step_buffer
        )
        state, action = self.n_step_buffer[0][:2]

        transition = (state, action, reward, next_state, done)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

        return self.n_step_buffer[0]

    def sample(self, batch_size, device):
        """Sample from the buffer random world"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.tensor(state, device=device, dtype=torch.float)
        action = torch.tensor(action, device=device).unsqueeze(1)
        reward = torch.tensor(reward, device=device, dtype=torch.float)
        next_state = torch.tensor(next_state, device=device, dtype=torch.float)
        done = torch.tensor(np.float32(done), device=device)

        return state, action, reward, next_state, done, indices

    def __len__(self):
        return len(self.buffer)

    def sample_batch_from_idxs(self, indices: np.ndarray, device) -> Dict[str, np.ndarray]:
        """Return the transition at indices"""
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done =  zip(*batch)
        
        state = torch.tensor(state, device=device, dtype=torch.float)
        action = torch.tensor(action, device=device).unsqueeze(1)
        reward = torch.tensor(reward, device=device, dtype=torch.float)
        next_state = torch.tensor(next_state, device=device, dtype=torch.float)
        done = torch.tensor(np.float32(done), device=device)

        return state, action, reward, next_state, done
     
    def _get_n_step_info(self, gamma, n_step_buffer: Deque) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rewards, next state, and done."""
        # info of the last transition
        rew, next_st, done = n_step_buffer[-1][-3:]

        # We reverse the deque to have the latest input first
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            next_st, done = (n_s, d) if d else (next_st, done)

        return rew, next_st, done

