'''
Disclaimer: The trpo part highly rely on trpo_mpi at @openai/baselines
'''

import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.gail.statistics import stats

def traj_segment_generator_composed(pi_task1, pi_task2, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def hybrid_learn(env, policy_func, reward_giver, rank,
          *,
          policy_step, boundary_step, entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name_1, task_name_2,
          gamma, lam,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,
          callback=None
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi_task1 = policy_func("pi_task1", ob_space, ac_space)
    oldpi_task1 = policy_func("oldpi_task1", ob_space, ac_space)

    pi_task2 = policy_func("pi_task2", ob_space, ac_space)
    oldpi_task2 = policy_func("oldpi_task2", ob_space, ac_space)

    atarg_task1 = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret_task1 = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    atarg_task2 = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret_task2 = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew_task1 = oldpi_task1.pd.kl(pi_task1.pd)
    ent_task1 = pi_task1.pd.entropy()
    meankl_task1 = tf.reduce_mean(kloldnew_task1)
    meanent_task1 = tf.reduce_mean(ent_task1)
    entbonus_task1 = entcoeff_task1 * meanent_task1

    kloldnew_task2 = oldpi_task2.pd.kl(pi_task2.pd)
    ent_task2 = pi_task2.pd.entropy()
    meankl_task2 = tf.reduce_mean(kloldnew_task2)
    meanent_task2 = tf.reduce_mean(ent_task2)
    entbonus_task2 = entcoeff_task2 * meanent_task2

    vferr_task1 = tf.reduce_mean(tf.square(pi_task1.vpred - ret_task1))
    vferr_task2 = tf.reduce_mean(tf.square(pi_task2.vpred - ret_task2))

    ratio_task1 = tf.exp(pi_task1.pd.logp(ac) - oldpi_task1.pd.logp(ac))  # advantage * pnew / pold
    ratio_task2 = tf.exp(pi_task2.pd.logp(ac) - oldpi_task2.pd.logp(ac))  # advantage * pnew / pold


    surrgain_task1 = tf.reduce_mean(ratio_task1 * atarg_task1)
    surrgain_task2 = tf.reduce_mean(ratio_task2 * atarg_task2)

    optimgain_task1= surrgain_task1 + entbonus_task1
    optimgain_task2 = surrgain_task2 + entbonus_task2

    optimgain = optimgain_task1 + optimgain_task2
    meankl = meankl_task1 +  meankl_task2
    entbonus = entbonus_task1 + entbonus_task2
    surrgain = surrgain_task1 + surrgain_task2
    meanent = meanent_task1 + meanent_task2

    losses_task1 = [optimgain_task1, meankl_task1, 
            entbonus_task1, surrgain_task1, meanent_task1]
    losses_task2 = [optimgain_task2, meankl_task2, 
            entbonus_task2, surrgain_task2, meanent_task2]

    # losses = [optimgain_task1, optimgain_task2, meankl_task1, meankl_task2, 
    #         entbonus_task1, entbonus_task2, surrgain_task1, surrgain_task2,
    #         meanent_task1, meanent_task2]
    loss_names = ['optimgain_task1', 'optimgain_task2', 'meankl_task1', 'meankl_task2', 
            'entbonus_task1', 'entbonus_task2', 'surrgain_task1', 'surrgain_task2',
            'meanent_task1', 'meanent_task2']

    dist_task1 = meankl_task1
    dist_task2 = meankl_task2

    all_var_list_task1 = pi_task1.get_trainable_variables()
    var_list_task1 = [v for v in all_var_list_task1 if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list_task1 = [v for v in all_var_list_task1 if v.name.startswith("pi/vff")]
    assert len(var_list_task1) == len(vf_var_list_task1) + 1
    # d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam_task1 = MpiAdam(vf_var_list_task1)

    all_var_list_task2 = pi_task2.get_trainable_variables()
    var_list_task2 = [v for v in all_var_list_task2 if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list_task2 = [v for v in all_var_list_task2 if v.name.startswith("pi/vff")]
    assert len(var_list_task2) == len(vf_var_list_task2) + 1
    # d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam_task2 = MpiAdam(vf_var_list_task2)

    get_flat_task1 = U.GetFlat(var_list_task1)
    set_from_flat_task1 = U.SetFromFlat(var_list_task1)
    klgrads_task1 = tf.gradients(dist_task1, var_list_task1)
    flat_tangent_task1 = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan_task1")
    shapes_task1 = [var.get_shape().as_list() for var in var_list_task1]
    start = 0
    tangents_task1 = []
    for shape in shapes_task1:
        sz = U.intprod(shape)
        tangents_task1.append(tf.reshape(flat_tangent_task1[start:start+sz], shape))
        start += sz
    gvp_task1 = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads_task1, tangents_task1)])  # pylint: disable=E1111
    fvp_task1 = U.flatgrad(gvp_task1, var_list_task1)

    get_flat_task2 = U.GetFlat(var_list_task2)
    set_from_flat_task2 = U.SetFromFlat(var_list_task2)
    klgrads_task2 = tf.gradients(dist_task2, var_list_task2)
    flat_tangent_task2 = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan_task2")
    shapes_task2 = [var.get_shape().as_list() for var in var_list_task2]
    start = 0
    tangents_task2 = []
    for shape in shapes_task2:
        sz = U.intprod(shape)
        tangents_task2.append(tf.reshape(flat_tangent_task2[start:start+sz], shape))
        start += sz
    gvp_task2 = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads_task2, tangents_task2)])  # pylint: disable=E1111
    fvp_task2 = U.flatgrad(gvp_task2, var_list)_task2

    assign_old_eq_new_task1 = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi_task1.get_variables(), pi_task1.get_variables())])
    compute_losses_task1 = U.function([ob, ac, atarg_task1], losses_task1)
    compute_lossandgrad_task1 = U.function([ob, ac, atarg_task1], losses_task1 + [U.flatgrad(optimgain_task1, var_list_task1)])
    compute_fvp_task1 = U.function([flat_tangent_task1, ob, ac, atarg_task1], fvp_task1)
    compute_vflossandgrad_task1 = U.function([ob, ret_task1], U.flatgrad(vferr_task1, vf_var_list_task1))

    compute_losses_task2 = U.function([ob, ac, atarg_task2], losses_task2)
    compute_lossandgrad_task2 = U.function([ob, ac, atarg_task2], losses_task2 + [U.flatgrad(optimgain_task2, var_list_task2)])
    compute_fvp_task2 = U.function([flat_tangent_task2, ob, ac, atarg_task2], fvp_task2)
    compute_vflossandgrad_task2 = U.function([ob, ret_task2], U.flatgrad(vferr_task2, vf_var_list_task2))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init_task1 = get_flat_task1()
    MPI.COMM_WORLD.Bcast(th_init_task1, root=0)
    set_from_flat_task1(th_init_task1)
    # d_adam.sync()
    vfadam_task1.sync()

    th_init_task2 = get_flat_task2()
    MPI.COMM_WORLD.Bcast(th_init_task2, root=0)
    set_from_flat_task2(th_init_task2)
    # d_adam.sync()
    vfadam_task2.sync()
    if rank == 0:
        print("Init param sum", th_init_task1.sum(), flush=True)
        print("Init param sum", th_init_task2.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_composed(pi_task1, pi_task2, env, reward_giver, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    g_loss_stats = stats(loss_names)
    # d_loss_stats = stats(reward_giver.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])

    fname = os.path.join(ckpt_dir, task_name)
    weight_file = tf.train.latest_checkpoint(ckpt_dir)

    print("fname: {} weight_file: {}".format(fname, weight_file))
    if weight_file is not None:
        U.load_state(weight_file)#, var_list=pi.get_variables())
        tf.logging.info('%s loaded' % weight_file)
    else:
        print("from scratch")
        tf.logging.info('Training from the scratch (no pre-trained weight_filets)..')

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)

            print("============= SAVE ===============")

        logger.log("********** Iteration %i ************" % iters_so_far)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")

        # First, we optimize each policy. and then next step we find optimized boundary for given policy.
        for _ in range(policy_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new()  # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                             include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)

        g_losses = meanlosses
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g), d_stepsize)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()
def policy_learn(env, policy_func, rank,
          *,
          entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,
          callback=None
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    assert len(var_list) == len(vf_var_list) + 1
    # d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    # d_adam.sync()
    vfadam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    # g_loss_stats = stats(loss_names)
    # d_loss_stats = stats(reward_giver.loss_name)
    # ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
    # if provide pretrained weight

    fname = os.path.join(ckpt_dir, task_name)
    weight_file = tf.train.latest_checkpoint(ckpt_dir)

    print("fname: {} weight_file: {}".format(fname, weight_file))
    if weight_file is not None:
        U.load_state(weight_file)#, var_list=pi.get_variables())
        tf.logging.info('%s loaded' % weight_file)
    else:
        print("from scratch")
        tf.logging.info('Training from the scratch (no pre-trained weight_filets)..')

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)

            print("============= SAVE ===============")

        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()



def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
