from baselines.hybrid_boundary_seeking import models  # noqa
from baselines.hybrid_boundary_seeking.build_graph import build_act, build_train  # noqa
from baselines.hybrid_boundary_seeking.simple import learn, load  # noqa
from baselines.hybrid_boundary_seeking.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)