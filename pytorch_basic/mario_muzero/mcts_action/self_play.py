from package import mcts
from package.game import Game
from package.replay_buffer import ReplayBuffer
from package.node import Node
from package.muzero_config import MuZeroConfig
from package.network import SharedStorage, Network


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
  game = config.new_game()

  while not game.terminal() and len(game.history) < config.max_moves:
    # At the root of the search tree we use the representation function to
    # obtain a hidden state given the current observation.
    root = Node(0)
    current_observation = game.make_image(-1)
    mcts.expand_node(root, game.legal_actions(),
                network.initial_inference(current_observation))
    mcts.add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    mcts.run_mcts(config, root, game.action_history(), network)
    action = mcts.select_action(config, len(game.history), root, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game

