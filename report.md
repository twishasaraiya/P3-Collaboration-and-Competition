## Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Architecture

```python
state_size = 24
action_size = 2
```

1. Actor

- Batch Normalization
- Linear(24, 512)
- relu
- Batch Normalization
- Linear(512,256)
- relu
- Batch Normalization
- Linear(256,2)
- tanh

2. Critic

- Batch Normalization
- Linear(24,512)
- Batch Normalization
- relu
- Linear(512+action_size,512)
- Batch Normalization
- relu
- Linear(512,1)

## Hyperparameter

BUFFER_SIZE = int(2e6)  # replay buffer size
BATCH_SIZE = 512       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.009             # for soft update of target parameters
LR_ACTOR = 0.0007         # learning rate of the actor
LR_CRITIC = 0.0007        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay


## Future work

The entire model is too fragile and seems to vary a lot based on the random seed. After playing with some random seed values the I observe the agent seems to learn and then after a point it starts degrading and the performance keeps on falling. The best result I got for RANDOM_SEED=9

Trying out other models like D4PG, AAC might produce better results.
