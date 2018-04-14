# HVHRL

Hybrid System Contol View of Hierarchical Reinforcement Learning

Contributors
: Kiwoo Shin, Saehong Park, Donggun Lee.

Environments:

(1) Python3.5
(2) openAI Gym
(3) Tensorflow 1.6 ( Latest version at this moment )

-----------------------------------------------------
Big picture:

(1) use dqn, not soft dqn
(2) iterative learning: 1) transition boundary learning, 2) optimal policy learning for composable policies.
(3) reward design

Work Flow:

(1) Pretrain each of composable policy and save policies as pkl files
  : set reward and environments for each of composable actions. 

  Done.

(2) Load pretrained policy from pkl file and test whether they run well.

  Done.

(3) Transition boundary learning: based on sampling and choose the best transition boundary to be used as supervised boundary learning.
  : random sample from trajectory, roll-out first and second policy, choose best transition points, use states before best transition states as it belongs to first policy, and rest of it as it belongs to second policy.
  : Use classification based method. <= This should be modified, but currently I cannot find any better method. There could be hint in policy gradient method and Deep Discovery of Option. But I don't know currently. Will find it.

  by 7pm

(4) Given the best transition boundary between two policies, optimize each policy with newly given reward function and some additive GAN loss to imitate each other near the boundary.

  by 10pm
