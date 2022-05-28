import torch
from pathlib import Path
import datetime

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

# Gym is an OpenAI toolkit for RL
from gym.wrappers import FrameStack
from mario_package.mario_env_handle import (SkipFrame,
                                            GrayScaleObservation,
                                            ResizeObservation)

from mario_package.mario_agent_learn import Mario
from mario_package.mario_logger import MetricLogger

# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)


use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / \
    datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84),
              action_dim=env.action_space.n,
              save_dir=save_dir,
              exploration_rate_decay=0.999999,  # 3000k step
              save_every=2e5,
              learn_rate=0.0005,  # double batch size double learning rate
              memory_maxlen=50000,  # to fit gpu memory
              batch_size=64  # to utilise gpu more by larger bbatch
              )
mario.load(r"C:\Users\Junhong Chen\Documents\GitHub\deep_rl_exercise\pytorch_basic\mario\checkpoints\2022-05-07T09-37-15\mario_net_5.chkpt")

logger = MetricLogger(save_dir)

total_step = 3600000
step_counter = 0
episode_counter = 0
while step_counter <= total_step:

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        step_counter += 1

        # Check if end of game
        if done or info["flag_get"]:
            episode_counter += 1
            break

    logger.log_episode()

    if episode_counter % 20 == 0:
        logger.record(episode=episode_counter, epsilon=mario.exploration_rate,
                      step=mario.curr_step)
