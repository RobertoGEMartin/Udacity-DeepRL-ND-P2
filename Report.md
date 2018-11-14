
# Report: "Project " - Continous Control"

We will train a DeepRL agent to solve a Unity Environment.

## Architecture

+ This image represents the flow of processes in a reinforcement learning training cycle.

![arch-rl](./img/arch-rl.png "arch-rl")


+ In this project, we use Unity like environment simulator engine and we use the PyTorch framework to build the deep RL agent.

![arch-deeprl-unity](./img/arch-deeprl-unity-2.png "arch-deeprl-unity")


+ The next image defines the block diagram of ML-Agents toolkit for our sample environment. 
+ In our project, we use an unique agent.

![arch-unity-1.png](./img/arch-unity-1.png "arch-unity-1.png")


## Code

The code is written in PyTorch and Python 3.6.2.


Main Files:  

+  /aaps/Reacher.app
   This app will simulate the Unity environment.
+ ddpq_agent.py
   This code defines the ddpg agent.
+ model.py
   This code defines de model of Neural Network architecture.
+ Continuous_Control_Rober.ipynb
   This notebook will train the agent.
+ /cp/
   Saved model weights of the successful agent.

## Learning Algorithm

We implement an artificial agent, termed [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)(DDPG)

DDPG is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

+ DDPG is an off-policy algorithm.
+ DDPG can only be used for environments with continuous action spaces.
+ DDPG can be thought of as being deep Q-learning for continuous action spaces.
+ DDPG can be implemented with parallelization

DDPG is a similarly foundational algorithm to VPG. DDPG is closely connected to Q-learning algorithms, and it concurrently learns a Q-function and a policy which are updated to improve each other.

Algorithms like DDPG and Q-Learning are off-policy, so they are able to reuse old data very efficiently. They gain this benefit by exploiting Bellman’s equations for optimality, which a Q-function can be trained to satisfy using any environment interaction data (as long as there’s enough experience from the high-reward areas in the environment).

### Pseudocode
![dpg-pseudocode](./img/ddpg-pseudocode.png "dpg-pseudocode")

### Hyper Parameters
#### DDPG Parameters

+ UFFER_SIZE = int(1e6)   # replay buffer size
+ BATCH_SIZE = 1024       # minibatch size
+ GAMMA = 0.99            # discount factor
+ TAU = 1e-3              # for soft update of target parameters
+ LR_ACTOR = 1e-3         # learning rate of the actor 
+ LR_CRITIC = 1e-3        # learning rate of the critic before: 3e-4
+ WEIGHT_DECAY = 0.0000   # L2 weight decay
+ EPSILON = 1.0           # noise factor
+ EPSILON_DECAY = 1e-6    # decay of noise factor

#### Neural Network. Model Architecture & Parameters.
For this project we use these models:

Actor Model:
  (bn0): BatchNorm1d(33, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)

Critic Model:
  (bn0): BatchNorm1d(33, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fcs1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)

### Training

### Plot of Rewards

Environment solved in 547 episodes. Average Score: 15.01

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +15.

![report-15.png - gif](./img/report-15-mac-cpu.png "report-15.png")

### Watch The DDPG Agent in Action

Video of The DDPG Agent:

![Video of Training](./videos/trained-dqn-agent-v2.gif "Video of Training")

[youtube video](https://youtu.be/_znGmJF6tKQ)
<!--- 
[![Trained DQN-Agent](http://img.youtube.com/vi/lBDV3A1hInQ/0.jpg)](http://www.youtube.com/watch?v=lBDV3A1hInQ "Trained DQN-Agent")
--->

### Ideas for Future Work

Future ideas for improving the agent's performance.

+ 
#### References

1. [Udacity Gihub Repo](https://github.com/udacity/deep-reinforcement-learning)
2. [Unity Docs](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md)
3. [Unity Paper](https://arxiv.org/abs/1809.02627)
4. [OpenAI master RL](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
5. [DDPG paper](https://arxiv.org/abs/1509.02971)
6. [OpenAI Baselines](https://blog.openai.com/better-exploration-with-parameter-noise/)
7. [Book: Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)
8. [PyTorch Agent Net: reinforcement learning toolkit for pytorch](https://github.com/Shmuma/ptan)
   