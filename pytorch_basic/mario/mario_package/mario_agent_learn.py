from .mario_agent_memory import Mario
import numpy as np
import torch

class Mario(Mario):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        
        current_state_Q = self.net(state, model="online")#Q for (batch,action_space)
        current_Q = current_state_Q[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        
        return current_Q

    @torch.no_grad() #don’t need to backpropagate on target (w,b)
    def td_target(self, reward, next_state, done):
        #use online model to find a'
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        #use target model to find q(s',a')
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

class Mario(Mario):
    def __init__(self, state_dim, action_dim, learn_rate):
        super().__init__(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learn_rate)
        self.loss_fn = torch.nn.SmoothL1Loss() #mean reduction

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target) #loss between current and target TD
        self.optimizer.zero_grad() #reset grad on each weight to 0, otherwise accumulate on batch
        loss.backward() #compute loss grad wrt weights
        #theta <- theta + lr * Delta(TD_target-q(s,a))
        self.optimizer.step() #adjust weights by grad and learning rate
        return loss.item() #report loss

    def sync_Q_target(self):
        #periodically copy online weights to target model
        self.net.target.load_state_dict(self.net.online.state_dict())


class Mario(Mario):
    def __init__(self, state_dim, action_dim, learn_rate, exploration_rate_decay, save_dir, save_every):
        super().__init__(state_dim, action_dim, learn_rate)
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync


        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        
        self.save_dir = save_dir
        self.save_every = save_every
    
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), #exploration rate changing during training
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):

        # increment step
        self.curr_step += 1
        
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory in batch
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate in batch
        td_est = self.td_estimate(state, action)

        # Get TD Target in batch
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return (td_est.mean().item(), loss)