from environment import Environment
from action import Action, ActionHistory
from node import Node
from typing import List
# A game is stored in the replay buffer when it is terminated
class Game(object):
  """A single episode of interaction with the environment."""
  #episode o0 a1 r1 o1 ... ot-1 at rt ot
  def __init__(self, action_space_size: int, discount: float):
    self.environment = Environment()  # Game specific environment.
    self.history = [] #action history [a1 a2 ...]
    self.rewards = [] #reward history [r1 r2 ...]
    self.child_visits = [] #searched policy [cv0 cv1 ...]
    self.root_values = [] #searched value [v0 v1 ...]
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    pass

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    return []

  def apply(self, action: Action):
    reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)

  def store_search_statistics(self, root: Node): #called after mcts to store searched value & policy
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    # returns (z0 r1 cv0), ..., (zt rt+1 cvt)
      # z0 = r1 + r2 + ... + v10
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      # For simplicity the network always predicts the most recently received
      # reward, even for the initial representation network where we already
      # know this reward.
      if current_index > 0 and current_index <= len(self.rewards):
        last_reward = self.rewards[current_index - 1]
      else:
        last_reward = 0

      if current_index < len(self.root_values):
        targets.append((value, last_reward, self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, last_reward, []))
    return targets

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)
