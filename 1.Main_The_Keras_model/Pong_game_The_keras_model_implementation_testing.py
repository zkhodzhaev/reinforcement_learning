# Zulfidin Khodzhaev


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
from rl.core import Processor

from PIL import Image
import gym
import numpy as np

# Constants
IMG_SHAPE = (84, 84)
WINDOW_LENGTH = 4  # Adjusted to 4 for efficiency
NB_STEPS = 1000000  # Total training steps
LEARNING_RATE = 0.00025  # Learning rate for Adam optimizer
GAMMA = 0.99  # Discount factor for future rewards
MEMORY_LIMIT = 1000000  # Memory limit for SequentialMemory


# the pong was giving out some random output, this is to get rid of those. But better function can be designed. 
class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomEnvWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, done, *extra_info = super().step(action)
        # Process extra_info to ensure it only contains simple, scalar values
        info = {}
        if extra_info:
            # Assuming extra_info is a list, iterate over it and add its elements to info
            for i, item in enumerate(extra_info):
                if isinstance(item, (int, float, bool, str)):
                    info[f'extra_{i}'] = item
                # Add more conditions here to handle other types if necessary
        return observation, reward, done, info
# Wrap your environment
env = CustomEnvWrapper(gym.make(game))
    
    
# Image processor
class ImageProcessor(Processor):
    def process_observation(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
            
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE).convert("L")
        return np.array(img).astype('uint8')
    
    def process_state_batch(self, batch):
        return batch / 255.0
    
    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

    
# Building the model
def build_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    # Convolutional layers
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_normal'))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    return model


# Environment setup
env = CustomEnvWrapper(gym.make("ALE/Pong-v5", render_mode = 'human'))
nb_actions = env.action_space.n
input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

# DQN Agent setup
model = build_model(input_shape, nb_actions)
memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGTH)
processor = ImageProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=500000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, processor=processor, 
               nb_steps_warmup=50000, gamma=GAMMA, target_model_update=10000, train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=LEARNING_RATE), metrics=['mae'])

# Load the pre-trained model
dqn.load_weights('Trained_weights.h5')

# Testing the agent
dqn.test(env, nb_episodes=1, visualize=False)
env.close()
