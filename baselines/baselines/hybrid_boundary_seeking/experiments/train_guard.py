import gym

from baselines import hybrid_boundary_seeking as deepq


'''

This script is to see whether learning boundary(guard) is possible.

Use two policies which are fixed and not trained over this script.
We only optimize guard model that can compose two policies together for new objective.

'''

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("Pendulum-v0")

    '''
    we assume that we have actor_list, which is a list of pre-trained policies 
    to be used as subpolicies
    '''
    subpolicies = []
    for actor in actor_list:
        actor = HBS.load("{}.pkl".format(actor))
        subpolicies.append(actor)

    guard_model = deepq.models.mlp([256,256])
    guard_act = deepq.learn(
        env,
        q_func=model,
        subpolicies=subpolicies,
        lr=1e-4,
        max_timesteps=400000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to pendulum_model.pkl")
    guard_act.save("pendulum_model.pkl")


if __name__ == '__main__':
    main()
