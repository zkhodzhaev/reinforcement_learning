# Zulfidin (Bobojon) Khodzhaev

#TESTING THE CODE

import numpy as np
import pickle
import gym

# Function definitions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    if isinstance(I, tuple):
        I = I[0]
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(float).ravel()

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state

# Load the trained model
model = pickle.load(open('save.p', 'rb'))

# Initialize the environment
env = gym.make("Pong-v0", render_mode = 'human')
observation = env.reset()

# Parameters
D = 80 * 80  # input dimensionality: 80x80 grid
prev_x = None
render = True  # Set to False if you don't need to visualize the test

# Test the model
num_test_episodes = 10
for episode in range(num_test_episodes):
    observation = env.reset()
    reward_sum = 0
    done = False
    while not done:
        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, _ = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3

        # step the environment
        observation, reward, done, *info = env.step(action)
        reward_sum += reward

        if done:
            print(f"Test Episode {episode + 1}: Reward Total = {reward_sum}")
            break

env.close()