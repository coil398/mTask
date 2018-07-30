import gym
import gym_super_mario_bros.actions
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from tqdm import tqdm
import numpy as np
import os
import pickle


class Train:

    _button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOP':    0b00000000,
    }

    def __init__(self):
        self._env = SuperMarioBrosEnv()
        self._env.reset()
        self._x_position = self._env._get_x_position()
        self._y_position = self._env._get_y_position()
        self._time_left = self._env._get_time()
        self._movement = gym_super_mario_bros.actions.RIGHT_ONLY
        self._set_actions_space(self._movement)
        self._rewards = list()

        if os.path.exists('mario.npy'):
            self._q_table = np.load('mario.npy')
        else:
            self._q_table = np.zeros((10000, len(self._movement)))

        self._epsilon = 0.02
        self._alpha = 0.2 # learning rate.
        self._gamma = 0.99 # time discount rate.

    def _set_actions_space(self, actions):
        self._env.action_space = gym.spaces.Discrete(len(actions))
        self._action_map = {}
        for action, button_list in enumerate(actions):
            byte_action = 0
            for button in button_list:
                byte_action |= self._button_map[button]
                self._action_map[action] = byte_action

    def _update_positions(self):
        self._x_position = self._env._get_x_position()
        self._y_position = self._env._get_y_position()

    def _get_action(self):
        if np.random.uniform(0, 1) > self._epsilon:
            _x_position = self._x_position
            _y_position = self._y_position
            _action = np.argmax(self._q_table[_x_position])
            # _action = np.argmax(self._q_table[_x_position][_y_position])
        else:
            _action = np.random.random_integers(1, len(self._movement)-1)
        # print('action: ', _action)
        return int(_action)

    def step(self, _action):
        _selected_action = self._action_map[_action]
        return self._env.step(_selected_action)

    def _get_x_reward(self):
        _x_position = self._env._get_x_position()
        _reward = _x_position - self._x_position
        if _reward < -5 or _reward > 5:
            return 0
        elif _reward == 0:
                return -1
        return _reward

    def _get_y_reward(self):
        _y_position = self._env._get_y_position()
        _reward = _y_position - self._y_position
        return _reward

    def _get_x_y_reward(self):
        _y_position = self._env._get_y_position()
        _reward = _y_position - self._y_position
        if _reward < 0:
            return 0
        return _reward

    def _get_time_reward(self):
        _time_left = self._env._get_time()
        _reward = _time_left - self._time_left
        self._time_left = _time_left
        if _reward > 0:
            return 0
        return _reward

    def _get_death_reward(self):
        if self._env._get_is_dying() or self._env._get_is_dead():
            return -25
        return 0

    def _get_reward(self):
        _reward =  self._get_x_reward() + self._get_x_y_reward() + self._get_time_reward() + self._get_death_reward()
        return _reward

    def _update_q_table(self, _action):
        _x_position = self._env._get_x_position()
        _y_position = self._env._get_y_position()

        _next_max_q_value = max(self._q_table[_x_position])
        # _next_max_q_value = max(self._q_table[_x_position][_y_position])
        _q_value = self._q_table[self._x_position][_action]
        # _q_value = self._q_table[self._x_position][self._y_position][_action]

        _new_q_value = _q_value + self._alpha * (self._get_reward() + self._gamma * _next_max_q_value - _q_value)

        # print('q_value: ', _q_value)
        # print('new_q_value: ', _new_q_value)

        self._q_table[self._x_position][_action] = _new_q_value
        # self._q_table[self._x_position][self._y_position][_action] = _new_q_value

    def _finalize(self):
        np.save('mario.npy', self._q_table)
        with open('mario.pkl', mode='wb') as f:
            pickle.dump(self._rewards, f)
        self._env.close()

        # for i in range(10000):
        #     for j in range(200):
        #         for k in range(len(self._movement)):
        #             print(self._q_table[i][j][k])

    def run(self):

        try:

            for i_episode in tqdm(range(50000)):
                _total_reward = 0
                self._env.reset()

                for t in range(10000):
                    self._env.render()

                    self._update_positions()
                    _action = self._get_action()
                    self.step(_action)

                    self._update_q_table(_action)

                    if self._env._get_done():
                        print('Episode is done after {} steps.'.format(t+1))
                        break
                        self._rewards.append(_total_reward)
        except KeyboardInterrupt:
            pass

        finally:

            self._finalize()


def main():
    train = Train()
    train.run()


if __name__ == '__main__':
    main()
