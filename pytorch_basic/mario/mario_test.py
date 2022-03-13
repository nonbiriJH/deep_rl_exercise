import torch
import time

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

# Gym is an OpenAI toolkit for RL
from gym.wrappers import FrameStack
from mario_package.mario_env_handle import SkipFrame, GrayScaleObservation, ResizeObservation

from mario_package.mario_agent_base import Mario

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

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n)
mario.load("/Users/junhongchen/Documents/GitHub/deep_rl_exercise/pytorch_basic/mario/checkpoints/2022-03-13T01-35-05/mario_net_3.chkpt")

episodes = 3
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        env.render()
        time.sleep(.01)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break