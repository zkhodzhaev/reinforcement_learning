# Zulfidin Khodzhaev

savefile = 'keras_model1.h5'
game = "ALE/Pong-v5"


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


nb_actions = env.action_space.n

IMG_SHAPE = (84, 84)
WINDOW_LENGTH = 12
input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

train = True 

class ImageProcessor(Processor):
    def process_observation(self, observation):
        
        IMG_SHAPE = (84, 84)
        if isinstance(observation, tuple):
            observation = observation[0]
            
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE)
        img = img.convert("L")
        img = np.array(img)
        return img.astype('uint8')
    
    def process_state_batch(self, batch):
        processed_batch = batch / 255.0
        return processed_batch
    
    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)
    
def build_the_model(input_shape, nb_actions=6):
    model = Sequential()
    
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    # First convolutional layer with smaller filter size
    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))

    # Second convolutional layer
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))

    # Flatten the network
    model.add(Flatten())

    # Single dense layer
    model.add(Dense(128, activation='relu'))

    # Output layer
    model.add(Dense(nb_actions, activation='linear'))
    
    return model


policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 
                              attr='eps', 
                              value_max=1.0, 
                              value_min=0.1, 
                              value_test=0.05, 
                              nb_steps=750000)

model = build_the_model(input_shape, nb_actions=nb_actions)

memory = SequentialMemory(limit=10000000, window_length=WINDOW_LENGTH)

processor = ImageProcessor()

#load checkpoint
checkpoint_filename = savefile
checkpoint_callback = ModelIntervalCheckpoint(checkpoint_filename, interval=500)

try:
    model.load_weights(checkpoint_filename)
    print(f"Loaded {checkpoint_filename}")
except:
    print(f"No checkpoint file to load under {checkpoint_filename}")

#end of checkpoint
 
dqn = DQNAgent(model=model, 
               nb_actions=nb_actions, 
               policy=policy, 
               memory=memory, 
               processor=processor, 
               nb_steps_warmup=75000, 
               gamma=0.95, 
               target_model_update=5000, 
               train_interval=12, 
               delta_clip=1)

dqn.compile(Adam(learning_rate=0.0005), metrics=['mae'])

if train == True:
    metrics = dqn.fit(env, nb_steps=1500000, callbacks=[checkpoint_callback], log_interval=100000, visualize=False)
    dqn.test(env, nb_episodes=1, visualize=True)
    env.close()
    model.summary()