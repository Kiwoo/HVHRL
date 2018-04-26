import gym

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("Pendulum-v0")
    model = deepq.models.mlp([256,256])

    exp_name = 'half_up'

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=350000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        exp_name=exp_name,
        callback=callback
    )
    print("Saving model to pendulum_model.pkl")
    act.save("pendulum_model_{}.pkl".format(exp_name))


if __name__ == '__main__':
    main()
