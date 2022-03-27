import tensorflow as tf


def prepare_data(dataset, sample_size, batch_size):
    """Prepare the dataset defined by the sample and batch size"""
    # normalize image values
    ds = dataset.map(lambda s, a, r, n: (s / 255, a, r, n / 255))
    # one hot encode actions
    ds = ds.map(
        lambda s, a, r, n: (
            s,
            tf.where(tf.equal(a, 3), [1.0, 0.0], [0.0, 1.0]),
            r,
            n,
        )
    )

    ds = ds.cache()
    ds = ds.shuffle(int(sample_size / 2))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(int(batch_size * 2))

    return ds


def prepare_state(state):
    """Get the state for picking the next action in the right shape to fit through the model"""
    state = tf.convert_to_tensor(state)
    state = tf.cast(state, tf.float32)
    state = state / 128 - 1
    state = tf.expand_dims(state, axis=0)

    return state
