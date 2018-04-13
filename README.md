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
  
(2) Load pretrained policy from pkl file and test whether they run well.
(3) Transition boundary learning: based on sampling and choose the best transition boundary to be used as supervised boundary learning.
(4) Given 
