import gym
import time
from baselines import hybrid_boundary_seeking as HBS

def main():
    env = gym.make("Pendulum-v0")
    boundary = "boundary"
    actor_list = ["half_down", "half_up"]

    boundary_act = HBS.load("pendulum_boundary_model.pkl", boundary)

    sub_policies = []
    for actor in actor_list:
        print("=== Actor: {}".format(actor))
        actor = HBS.load("pendulum_model_{}.pkl".format(actor), actor)
        sub_policies.append(actor)    


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
