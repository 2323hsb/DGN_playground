import numpy as np
import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, input_dim=2, z_dims=4, name='vae'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.z_dims = z_dims

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.input_dim,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self.z_dims * 2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.z_dims,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self.input_dim),
            ]
        )

    def encode(self, x):
        z_mean, z_logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_logvar

    def decode(self, z):
        x_f = self.decoder(z)
        return x_f

    def sampling(self, x):
        z_mean, z_logvar = self.encode(x)
        z = z_mean + tf.exp(z_logvar) * tf.random.normal(tf.shape(z_mean), mean=0.0, stddev=1., dtype=tf.float32)
        x_f = self.decode(z)
        x_f = tf.clip_by_value(x_f, 0, 1)

        return z_mean, z_logvar, x_f