import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time
import random

world = random.randint(1, 8)
stage = random.randint(1, 4)

# env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    env.render()
    time.sleep(1.0 / 50)

env.close()
