from game import Game
from muzero_config import MuZeroConfig

#store episodes and generate batches for training
class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size #size of buffer
    self.batch_size = config.batch_size #number of games (episodes) drawn in sampling
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    # atari unrolls 5 steps
    # a sample in the batch represents ot at+1 rt+1 ... ot+4 at+5 rt+5 
    # returns (ot history target) per sample
    # history at+1 ... at+5
    # target: (zt rt+1 cvt), ... (zt+5 rt+6 cvt+5)
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps))
            for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return self.buffer[0]

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return -1
