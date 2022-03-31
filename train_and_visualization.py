import matplotlib.pyplot as plt
import tensorflow as tf


def choose_action(model, x, episode):
    """Choose action considering thompson sampling"""

    if episode < 500000:
        # high temperature (low value) produces nearly probabilities of 0.5
        temperature = 1
    else:
        episode = tf.cast(episode, tf.float32)
        # getting lower temperature (higher values) for later episodes
        # for more differentiated probabilities
        # starts with 1 for 500000 episodes
        # ends with 10 for 1000000 episodes
        # other values would need another scaling
        temperature = 0.000018 * episode - 8

    p = model(x)
    probs = tf.nn.softmax(p * temperature)  # thompson sampling
    # get action dependent on higher probability
    action = int(tf.random.categorical(probs, 1))

    return action + 3  # add 3 to get correct action index (3 = up, 4 = down)


def calculate_targets(q_values, rewards, gamma):
    """Calculate the Q target"""
    q_targets = rewards + gamma * tf.reduce_max(q_values)
    return q_targets


# train based on data samples from erb
def training(data, model, loss_function, optimizer, gamma):
    """Train the model for the number of epochs specified.

    Args:
        data:  dataset
        model: model to train.
        loss_function: loss function used for the training and test the model.
        optimizer:  optimizer for the train step.
        gamma:  discount factor for reward

    Returns:
        average training loss
    """

    for s, a, r, n in data:
        # calculate q values for next state
        q_values = model(n)
        # calculate target
        # target = reward for terminating steps
        # target = reward + gamma * argmax(q_values) else
        targets = tf.where(tf.equal(r, 0), calculate_targets(q_values, r, gamma), r)

        with tf.GradientTape() as tape:
            # make predictions for current state
            q_preds = model(s)
            # calculate loss between taget and q prediction for the action
            loss = loss_function(targets, tf.reduce_sum(q_preds * a, axis=-1))
            # update gradient and optimizer
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def visualize(losses, rewards):
    """Prepares the visualization on two plots in a single subplot
       displaying the average losses and rewards.

    Args:
       losses: average of all the losses for each epoch
       rewards: average of all the rewards for each epoch
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 6))
    ax[0].plot(losses, label="loss")
    ax[1].plot(rewards, label="reward")
    ax[0].set(ylabel="Average loss", title="Learning progress over epochs")
    ax[1].set(ylabel="Average reward")
    plt.xlabel("Epochs")
    plt.tight_layout()
    plt.show()
