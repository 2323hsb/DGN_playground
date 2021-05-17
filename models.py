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
        z_mean, z_logvar = tf.split(
            self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_logvar

    def decode(self, z):
        x_f = self.decoder(z)
        return x_f

    def sampling(self, x):
        z_mean, z_logvar = self.encode(x)
        z = z_mean + tf.exp(z_logvar) * tf.random.normal(tf.shape(z_mean),
                                                         mean=0.0, stddev=1., dtype=tf.float32)
        x_f = self.decode(z)
        x_f = tf.clip_by_value(x_f, 0, 1)

        return z_mean, z_logvar, x_f


class CVAE(tf.keras.Model):
    def __init__(self, input_dim=2, cond_dim=2, z_dims=4, name='cvae'):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.z_dims = z_dims

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.input_dim+self.cond_dim,)),
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
                tf.keras.layers.Input(shape=(self.z_dims+self.cond_dim,)),
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

    def encode(self, x_cond):
        x_cond = tf.concat([x_cond[0], x_cond[1]], axis=1)
        z_mean, z_logvar = tf.split(self.encoder(
            x_cond), num_or_size_splits=2, axis=1)
        return z_mean, z_logvar

    def decode(self, z, cond):
        z_cond = tf.concat([z, cond], axis=1)
        x_f = self.decoder(z_cond)
        return x_f

    def sampling(self, x_cond):
        z_mean, z_logvar = self.encode(x_cond)
        z = z_mean + tf.exp(z_logvar) * tf.random.normal(tf.shape(z_mean),
                                                         mean=0.0, stddev=1., dtype=tf.float32)
        x_f = self.decode(z, x_cond[1])
        x_f = tf.clip_by_value(x_f, 0, 1)

        return z_mean, z_logvar, x_f


class GAN(tf.keras.Model):
    def __init__(self, noise_dim=64, target_dim=2, name='gan'):
        super(GAN, self).__init__()
        self.noise_dim = noise_dim
        self.target_dim = target_dim

        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.noise_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.target_dim),
            ]
        )

        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.target_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ]
        )

    def generate(self, noise):
        x_f = self.generator(noise)
        return x_f

    def discriminate(self, x):
        prob = self.discriminator(x)
        return prob


class CVAEGAN(tf.keras.Model):
    def __init__(self, input_dim=2, z_dim=4, cond_dim=2, name='cvaegan'):
        super(CVAEGAN, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.input_dim+self.cond_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.z_dim*2),
            ]
        )

        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.z_dim+self.cond_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.input_dim),
            ]
        )

        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.input_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(64, activation='relu', name='f_d'),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ]
        )

        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.input_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(64, activation='relu', name='f_c'),
                tf.keras.layers.Dense(self.cond_dim, activation='softmax'),
            ]
        )

    def encode(self, x_cond):
        x_cond = tf.concat([x_cond[0], x_cond[1]], axis=1)
        z_mean, z_logvar = tf.split(self.encoder(
            x_cond), num_or_size_splits=2, axis=1)
        return z_mean, z_logvar

    def generate(self, z, cond):
        z_cond = tf.concat([z, cond], axis=1)
        x_f = self.generator(z_cond)
        return x_f

    def discriminate(self, x):
        inp = self.discriminator.input
        f_D_out = self.discriminator.get_layer('f_d').output
        f_D_model = tf.keras.Model(inputs=inp, outputs=f_D_out)
        return self.discriminator(x), f_D_model(x)

    def classify(self, x):
        inp = self.classifier.input
        f_C_out = self.classifier.get_layer('f_c').output
        f_C_model = tf.keras.Model(inputs=inp, outputs=f_C_out)
        return self.classifier(x), f_C_model(x)

    def sampling(self, x_cond):
        z_mean, z_logvar = self.encode(x_cond)
        z = z_mean + tf.exp(z_logvar) * tf.random.normal(tf.shape(z_mean),
                                                         mean=0.0, stddev=1., dtype=tf.float32)
        x_f = self.generate(z, x_cond[1])
        x_f = tf.clip_by_value(x_f, 0, 1)

        return z_mean, z_logvar, x_f
