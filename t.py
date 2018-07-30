import gym

env = gym.make('CartPole-v0')
total_reward_sum = [0] * 20
for i_episode in range(20):
    observation = env.reset()
    total_reward = 0
    for t in range(100):
        env.render()
        print(observation)
        # action = env.action_space.sample()
        sum_obs = observation[0] + observation[1] + observation[2] + observation[3]
        if sum_obs > 0:
            action = 1
        else:
            action = 0
        observation, reward, done, info = env.step(action)
        total_reward += 1
        if done:
            print('Episode finished after {} timesteps.'.format(t+1))
            print('total reward is', total_reward)
            total_reward_sum[i_episode] = total_reward
            if i_episode == 19:
                print('all total reward is', total_reward_sum)
            break
        if t == 99:
            print('Episode finished after {} timesteps.'.format(t+1))
            print('total reward is', total_reward)
            total_reward_sum[i_episode] = total_reward
            if i_episode == 19:
                print('all total reward is', total_reward_sum)
