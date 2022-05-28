from .mario_agent_base import Mario
import torch
from collections import deque
import random

#agent's memory
class Mario(Mario):  # subclassing for continuity
    def __init__(self, state_dim, action_dim, memory_maxlen, batch_size):
        super().__init__(state_dim, action_dim)
        self.memory = deque(maxlen=memory_maxlen)
        self.batch_size = batch_size

    def cache(self, state, next_state, action, reward, done):
        """
        Each time Mario performs an action, he stores the experience to his memory. 
        His experience includes the current state, action performed, reward from the action, the next state, and whether the game is done.
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        # *batch: unpact batch to individual (s,s',a,r,d)
        # zip(): create an iteracble on unpacked batch
        # map(): apply torch.stack in each iterable
        # torch.stack: Concatenates a sequence of tensors along a new dimension.
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        # squeeze(): dimension 1 is removed
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()