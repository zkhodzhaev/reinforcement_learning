

# Reinforcement Learning: Pong Game Implementations

 Why?: Keras implementation of a blog post (Deep Reinforcement Learning: Pong from Pixels) that originally used Python's numpy library for neural network operations.

During testing, the trained model was executed using pure Python in Jupyter Notebook:

<img src="https://github.com/zkhodzhaev/reinforcement_learning/assets/21960382/466749f6-33ca-4a2e-9509-8da9ed0e52a9" width="400" height="250">

### Next, this model was implemented using Keras and TensorFlow. Here, I show the model and its one-run reward 
#### (Note: these models are run < day. More training will improve performance):

The Model that performed the best:

<img width="1094" alt="image" src="https://github.com/zkhodzhaev/reinforcement_learning/assets/21960382/63938ff6-66ed-4c16-affe-da680488def0">





### Other Models:
Model 1:

<img width="422" alt="image" src="https://github.com/zkhodzhaev/reinforcement_learning/assets/21960382/1b9f80b9-aecb-424c-b8ff-4d3d5d969749">


Model 2:

<img width="428" alt="image" src="https://github.com/zkhodzhaev/reinforcement_learning/assets/21960382/8c00cadf-fb0e-41b0-b238-136d9c226f2e">



# More details:

This repository contains two implementations of a Pong game agent - one in pure Python and one using Keras/TensorFlow. The goal is to contrast different approaches to this reinforcement learning problem.

#### Python Implementation

The Python implementation in Pong_game_python_implementation.py follows a policy gradient method using RMSProp and a simple 2-layer neural network model. Key aspects:

State Representation: The Pong screen is preprocessed into an 80x80 1D vector
Model Architecture: 2 fully-connected hidden layers, using ReLU activations and Xavier initialization
Training: Policy gradient using discounted rewards and RMSProp parameter updates
Keras Implementation
The Keras implementation in Pong_game_The_keras_model.py uses a Deep Q Learning (DQN) agent with a Convolutional Neural Network (CNN) model. Key aspects:

State Representation: Stores last 4 frames as 84x84 grayscale images
Model Architecture: 3 Conv2D layers for feature extraction, followed by fully-connected layers
Training: DQN agent with experience replay, target network, and ε-greedy exploration strategy
The CNN automatically extracts spatial features from the game screen, instead of manual engineering of the input. Training is done through Q-learning updates based on memories of (state, action, reward) transitions.

