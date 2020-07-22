import gym

from stable_baselines3 import PPO



if __name__ == '__main__':

    env = gym.make('LunarLander-v2')

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log= "./runs")
    model.learn(total_timesteps=1000000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
