# P3-Collaboration-and-Competition

![](https://github.com/twishasaraiya/P3-Collaboration-and-Competition/blob/master/assets/tennis.gif)
## Framework 

- PyTorch
- Python 3

## The Environment

For this project, you will work with the Tennis environment.

[Tennis Playing Agents](https://github.com/twishasaraiya/P3-Collaboration-and-Competition/blob/master/tennis.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

1. For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

  Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

  Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

  (For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

  (For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

2. Download Anaconda from [here](https://www.anaconda.com/distribution/)
3. Create a new virtualenv
    
    ```python
    conda create -n [env name] python=3.6
    ```
4. Download the zip file and extract the files in folder of your choice
5. Activate the environment
```python
conda activate [env name]
```
6. cd into the folder `cd folder_name/` and start jupyter notebook
```
(env_name)$cd folder_name
(env_name)folder_name$ jupyter notebook
```
7. Follow the instructions in `Tennis.ipynb` to train your own agent
