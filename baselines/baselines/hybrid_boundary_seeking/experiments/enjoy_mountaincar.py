import gym

from baselines import hybrid_boundary_seeking as HBS


def main():
    env = gym.make("MountainCar-v0")
    act = HBS.load("mountaincar_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
