import gym
import time
from baselines import hybrid_boundary_seeking as HBS


def main():
    env = gym.make("Pendulum-v0")
    act = HBS.load("cartpole_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.01)
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
