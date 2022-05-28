import torch
from .mario_model import MarioNet
import numpy as np


class Mario:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = MarioNet(self.state_dim, self.action_dim).float()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1

    def load(self, load_path):
        try:
            self.net.load_state_dict(torch.load(load_path)['model'])
            self.exploration_rate = torch.load(load_path)['exploration_rate']
        except:
            print(
                f"no weights are loaded as either {load_path} cannot be found or incompatible to current model.")
        else:
            print(
                f"weights are loaded successfuly! exploration_rate is {self.exploration_rate}")

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()  # from lazy frame to array
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        return action_idx
