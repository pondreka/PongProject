import matplotlib.pyplot as plt
import tensorflow as tf


def choose_action(model, x, epoch, max_epoch):
    """Choose action considering thompson sampling"""

    if epoch < int(max_epoch / 2):
        # high temperature (low value) produces nearly probabilities of 0.5
        temperature = 1
    else:
        epoch = tf.cast(epoch, tf.float32)
        # getting lower temperature (higher values) for later episodes
        # for more differentiated probabilities
        # starts with 1.5 for 10000 epochs
        # ends with 11 for 20000 epochs
        # more epochs would need another scaling
        temperature = 0.2 * tf.math.exp(0.0002 * epoch)

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
    loss_aggregator = []

    for s, a, r, n in data:
        # calculate q values for next state
        q_values = model(n)
        # calculate target by target = reward + gamma * argmax(q_values)
        targets = calculate_targets(q_values, r, gamma)

        with tf.GradientTape() as tape:
            # make predictions for current state
            q_preds = model(s)
            # calculate loss between taget and q prediction for the action
            loss = loss_function(targets, tf.reduce_sum(q_preds * a, axis=-1))
            loss_aggregator.append(loss.numpy())
            # update gradient and optimizer
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    test_loss = tf.reduce_mean(loss_aggregator)

    return test_loss


#
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
