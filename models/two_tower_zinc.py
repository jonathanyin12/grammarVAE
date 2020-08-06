import copy
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, Reshape, Flatten, RepeatVector, TimeDistributed, GRU, Convolution1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import zinc_grammar as G
import h5py
from tensorflow.python.keras import optimizers
import json


masks_K = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

MAX_LEN = 277
MAX_LEN_FUNCTIONAL = 200
DIM = G.D


class MoleculeVAE():
    autoencoder = None

    def create(self,
               charset,
               max_length=MAX_LEN,
               max_length_functional=MAX_LEN_FUNCTIONAL,
               latent_rep_size=2,
               weights_file=None,
               load_optimizer=False):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length))
        f = Input(shape=(max_length_functional, 1))

        _, z = self._buildEncoder(x, f, latent_rep_size, max_length, max_length_functional)
        self.encoder = Model([x, f], z)

        encoded_input = Input(shape=(latent_rep_size,))
        o, fo = self._buildDecoder(
            encoded_input,
            latent_rep_size,
            max_length,
            charset_length
        )
        self.decoder = Model(
            encoded_input,
            [o, fo]
        )

        x1 = Input(shape=(max_length, charset_length))
        f1 = Input(shape=(max_length_functional, 1))
        vae_loss, z1 = self._buildEncoder(x1, f1, latent_rep_size, max_length, max_length_functional)
        o1, fo1 = self._buildDecoder(
            z1,
            latent_rep_size,
            max_length,
            charset_length
        )
        self.autoencoder = Model(
            [x1, f1],
            [o1, fo1]
        )

        # for obtaining mean and log variance of encoding distribution
        x2 = Input(shape=(max_length, charset_length))
        f2 = Input(shape=(max_length_functional, 1))
        (z_m, z_l_v) = self._encoderMeanVar(x2, f2, latent_rep_size, max_length, max_length_functional)
        self.encoderMV = Model(inputs=[x2, f2], outputs=[z_m, z_l_v])


        if weights_file:
            temp_autoencoder = load_model(weights_file, custom_objects={'vae_loss': vae_loss})
            trainable_variables = temp_autoencoder.trainable_variables

            opened_new_file = not isinstance(weights_file, h5py.File)
            if opened_new_file:
                f = h5py.File(weights_file, mode='r')
            else:
                f = weights_file

            training_config = json.loads(f.attrs.get('training_config'))

            def load_attributes_from_hdf5_group(group, name):
                """Loads attributes of the specified name from the HDF5 group.
                This method deals with an inherent problem
                of HDF5 file which is not able to store
                data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
                Arguments:
                    group: A pointer to a HDF5 group.
                    name: A name of the attributes to load.
                Returns:
                    data: Attributes data.
                """
                if name in group.attrs:
                    data = [n.decode('utf8') for n in group.attrs[name]]
                else:
                    data = []
                    chunk_id = 0
                    while '%s%d' % (name, chunk_id) in group.attrs:
                        data.extend(
                            [n.decode('utf8') for n in group.attrs['%s%d' % (name, chunk_id)]])
                        chunk_id += 1
                return data

            def load_optimizer_weights_from_hdf5_group(hdf5_group):
                """Load optimizer weights from a HDF5 group.
                Arguments:
                    hdf5_group: A pointer to a HDF5 group.
                Returns:
                    data: List of optimizer weight names.
                """
                weights_group = hdf5_group['optimizer_weights']
                optimizer_weight_names = load_attributes_from_hdf5_group(
                    weights_group, 'weight_names')
                return [weights_group[weight_name] for weight_name in optimizer_weight_names]
            optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)

            optimizer_config = training_config['optimizer_config']
            optimizer = optimizers.deserialize(optimizer_config)
            _ = optimizer.iterations
            optimizer._create_hypers()
            optimizer._create_slots(trainable_variables)
            optimizer.set_weights(optimizer_weight_values)

            self.autoencoder.load_weights(weights_file)

            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)
            self.encoderMV.load_weights(weights_file, by_name=True)
            if load_optimizer:
                self.autoencoder.compile(optimizer=optimizer,
                                         loss={'decoded_mean': vae_loss, 'decoded_mean_2': vae_loss})
                print("learning rate: ", K.eval(self.autoencoder.optimizer.lr))
                print("Optimizer weights: ", [K.eval(w) for w in self.autoencoder.optimizer.weights])
            else:
                self.autoencoder.compile(optimizer=Adam(learning_rate=5e-4),
                                         loss={'decoded_mean': vae_loss, 'decoded_mean_2': vae_loss})
        else:
            self.autoencoder.compile(optimizer=Adam(learning_rate=5e-4),
                                     loss={'decoded_mean': vae_loss, 'decoded_mean_2': vae_loss})

    # Encoder tower structure
    def _towers(self, x, f):
        # Tower 1
        h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)

        # Tower 2
        hf = Flatten(name='tower_2_flatten_enc')(f)
        hf = Dense(256, activation='relu', name='tower_2_dense_1')(hf)

        # Merge
        h = Concatenate()([h, hf])
        return Dense(512, activation='relu', name='dense_1')(h)

    def _encoderMeanVar(self, x, f, latent_rep_size, max_length, max_length_func):
        h = self._towers(x, f)

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        return z_mean, z_log_var

    def _buildEncoder(self, x, f, latent_rep_size, max_length, max_length_func, epsilon_std=0.01):
        h = self._towers(x, f)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        # this function is the main change.
        # essentially we mask the training data so that we are only allowed to apply
        #   future rules based on the current non-terminal
        def conditional(x_true, x_pred):
            most_likely = K.argmax(x_true)
            most_likely = tf.reshape(most_likely, [-1])  # flatten most_likely
            ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely), 1)  # index ind_of_ind with res
            ix2 = tf.cast(ix2, tf.int32)  # cast indices as ints
            M2 = tf.gather_nd(masks_K, ix2)  # get slices of masks_K with indices
            M3 = tf.reshape(M2, [-1, MAX_LEN, DIM])  # reshape them
            P2 = tf.multiply(K.exp(x_pred), M3)  # apply them to the exp-predictions
            P2 = tf.divide(P2, K.sum(P2, axis=-1, keepdims=True))  # normalize predictions
            return P2

        def vae_loss(x, x_decoded_mean):
            if K.int_shape(x_decoded_mean)[1] == max_length:
                x_decoded_mean = conditional(x, x_decoded_mean)  # we add this new function to the loss
                x = K.flatten(x)
                x_decoded_mean = K.flatten(x_decoded_mean)
                xent_loss = max_length * binary_crossentropy(x, x_decoded_mean)
            elif K.int_shape(x_decoded_mean)[1] == max_length_func:
                x = tf.reshape(x, (-1, max_length_func))
                x_decoded_mean = tf.reshape(x_decoded_mean, (-1, max_length_func))
                xent_loss = mse(x, x_decoded_mean)
            else:
                raise ValueError('UNRECOGNIZED SHAPE')

            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

            return xent_loss + kl_loss

        return vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        l = Dense(latent_rep_size, name='latent_input', activation='relu')(z)

        # Tower 1
        h = RepeatVector(max_length, name='repeat_vector')(l)
        h = GRU(512, return_sequences=True, reset_after=True, name='gru_1')(h)
        h = GRU(512, return_sequences=True, reset_after=True, name='gru_2')(h)
        h = GRU(512, return_sequences=True, reset_after=True, name='gru_3')(h)
        h = TimeDistributed(Dense(charset_length), name='decoded_mean')(h)

        # Tower 2
        hf = Dense(200, name='dense_tower_2', activation='sigmoid')(l)
        hf = Reshape((200, 1), name='decoded_mean_2')(hf)

        return h, hf

    def save(self, filename):
        self.autoencoder.save_weights(filename)

    def load(self, charset, weights_file, latent_rep_size=128, max_length=MAX_LEN, load_optimizer=False):
        self.create(charset, max_length=max_length, weights_file=weights_file, latent_rep_size=latent_rep_size, load_optimizer=load_optimizer)