# Homework3-Policy Gradient

In this homework, you will use a neural network to learn a parameterize policy that can select action without consulting a value function. A value function may still be used to learn the policy weights, but is not required for action selection. 

There are some advantage of the policy-based algorithms:

- Policy-based methods also offer useful ways of dealing with continuous action spaces
- For some tasks, the policy function is simpler and thus easier to approximate.


## Introduction

We will use ```CartPole-v0``` as environment in this homework. The following gif is the visualization of the CartPole: 

<img src="https://cloud.githubusercontent.com/assets/7057863/19025154/dd94466c-8946-11e6-977f-2db4ce478cf3.gif" width="400" height="200" />

For further description, please see [here](https://gym.openai.com/envs/CartPole-v0)

## Setup
- Python 3.5.3
- OpenAI gym
- **tensorflow**
- numpy
- matplotlib
- ipython

We encourage you to install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html) in your laptop to avoid tedious dependencies problem.

**for lazy people:**

```
conda env create -f environment.yml
source activate cedl
# deactivate when you want to leave the environment
source deactivate cedl
```

## TODO

- [60%] Problem 1,2,3: Policy gradient 
- [20%] Problem 5: Baseline bootstrapping 
- [10%] Problem 6: Generalized Advantage Estimation
  - for lazy person, you can refer to [here](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/High-Dimensional%20Continuous%20Control%20Using%20Generalized%20Advantage%20Estimation.md)
- [10%] Report 
- [5%] Bonus, share you code and what you learn on github or  yourpersonal blogs, such as [this](https://andrewliao11.github.io/object/detection/2016/07/23/detection/)



## Other
- Deadline: 11/2 23:59, 2017
- Some of the codes are credited to Yen-Chen Lin :smile:
- Office hour 2-3 pm in 資電館711 with [Yuan-Hong Liao](https://andrewliao11.github.io).
- Contact *andrewliao11@gmail.com* for bugs report or any questions.
