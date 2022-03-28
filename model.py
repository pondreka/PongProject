import tensorflow as tf


class DQModel(tf.keras.Model):
    """A custom DQ model"""

    def __init__(self, hidden_size: int, num_actions: int = 2):
        """Constructor

        :param hidden_size:
        :param num_actions: number of resulting actions
        """
        super(DQModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(8, 8), strides=(4, 4), activation="relu"
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
        )

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            hidden_size,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )

        self.out = tf.keras.layers.Dense(
            num_actions,
            activation="tanh",  # change to linear for other games
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )

    def call(self, inputs):
        """Layer propagation
        Args:
            inputs
        Return:
            propagation results
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.out(x)
        return x
