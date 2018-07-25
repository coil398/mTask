from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from tqdm import tqdm
import time


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)


try:
    done = True
    for _ in tqdm(range(5000)):
        if done:
            time.sleep(3)
            state = env.reset()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print('state: ', state)
        print('reward: ', reward)
        env.render()
except KeyboardInterrupt:
    pass

env.close()
