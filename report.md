## Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Architecture

```python
state_size = 24
action_size = 2
```

1. Actor

bn1 => BatchNorm1d(24)
fc1 => Linear(24, 512)
        
bn2 => BatchNorm1d(512)
fc2 => Linear(512, 256)
        
bn3 => BatchNorm1d(256)
fc3 => Linear(256, 2)

2. Critic

bn1 => BatchNorm1d(24)
fcs1 => Linear(24, 512)
        
bn2 => BatchNorm1d(512)
fc2 => Linear(512+2, 512)
        
bn3 => BatchNorm1d(512)
fc3 => Linear(512, 1)

## Hyperparameter

BUFFER_SIZE = int(2e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 3e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay


## Future work

The entire model is too fragile and seems to vary a lot based on the random seed. After playing with some random seed values the I observe the agent seems to learn and then after a point it starts degrading and the performance keeps on falling after 500-600 episodes. 

Trying out other models like MADPPG, AAC might produce better results.
