import gym

from baselines import hybrid_boundary_seeking as HBS


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

    exp_name = 'boundary'

    actor_list = ["half_down", "half_up"]
    sub_policies = []
    for actor in actor_list:
        print("=== Actor: {}".format(actor))
        actor = HBS.load("pendulum_model_{}.pkl".format(actor), actor)
        sub_policies.append(actor)

    boundary_model = HBS.models.mlp([256,256])
    boundary_act = HBS.learn(
        env,
        q_func=boundary_model,
        sub_policies=sub_policies,
        lr=1e-4,
        max_timesteps=400000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        exp_name = exp_name,
        callback=callback
    )
    print("Saving model to pendulum_model.pkl")
    boundary_act.save("pendulum_model_{}.pkl".format(exp_name))


if __name__ == '__main__':
    main()
