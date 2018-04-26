import gym
import time
from baselines import deepq


def main():
    env = gym.make("Pendulum-v0")
    exp_name = 'half_down'
    act = deepq.load("pendulum_model_{}.pkl".format(exp_name), exp_name)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.01)
            print("obs: {}".format(obs[None]))
            print("act: {}".format(act(obs[None])[0]))
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
