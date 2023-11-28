# Zulfidin Khodzhaev


import numpy as np
import tensorflow as tf
import gym

# Hyperparameters
H = 20  # number of hidden layer neurons
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
D = 80 * 80  # input dimensionality: 80x80 grid
render = True

# Keras Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(H, activation='relu', input_shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Preprocessing function
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    if isinstance(I, tuple):
        I = I[0]
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(float).ravel()

# Discounted rewards function
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Custom loss function
def custom_loss(advantages, predicted_action_probs):
    # advantages: The discounted rewards
    # predicted_action_probs: The probabilities of the chosen actions from the model

    # Inverting the probabilities for actions not taken
    inverted_action_probs = 1 - predicted_action_probs

    # Combining the probabilities with the advantages
    loss = -tf.reduce_mean(tf.math.log(predicted_action_probs) * advantages + tf.math.log(inverted_action_probs) * (1 - advantages))
    return loss


# Training loop
env = gym.make("Pong-v0", render_mode='human')
observation = env.reset()

prev_x = None
xs, dlogps, drs = [], [], []
reward_sum = 0
episode_number = 0

running_reward = None 

while True:
    env.render() 
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob = model.predict(x.reshape(1, -1), batch_size=1).flatten()
    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x)
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)

    observation, reward, done, info = env.step(action)[:4]
    reward_sum += reward
    drs.append(reward)

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, dlogps, drs = [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= (np.std(discounted_epr) + 1e-10)

        with tf.GradientTape() as tape:
            p = model(epx, training=True)
            loss = custom_loss(discounted_epr, p)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('Resetting env. Episode reward total was %.f. Running mean: %.f' % (reward_sum, running_reward))
        reward_sum = 0
        observation = env.reset()
        prev_x = None

#         if episode_number % 100 == 0:
#             model.save('pong_model.h5')
        model.save('pong_model_local.h5')

    if reward != 0:
        print('Ep %d: Game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
