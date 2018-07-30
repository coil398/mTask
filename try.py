from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from tqdm import tqdm
import time


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)

print('action space: ', env.action_space)
print('observation space: ', env.observation_space)

for i_episode in tqdm(range(20)):
    observation = env.reset()
    for t in range(5000):
        env.render()
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(3)
        print('-1: ', observation[-1])
        print('reward: ', reward)
        import time; time.sleep(10)
        import sys; sys.exit()
        if done:
            print('Episode finished after {} steps.'.format(t+1))
            break


env.close()
