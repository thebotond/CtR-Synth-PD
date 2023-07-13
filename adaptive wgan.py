import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU device found: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU found. Please make sure you have installed the necessary dependencies correctly.")

# Set the GPU memory growth option to allow memory allocation as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load and preprocess the input data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Normalize numerical data using min-max scaling
def normalize_data(data):
    numerical_cols = data.select_dtypes(include=np.number).columns
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].min()) / (data[numerical_cols].max() - data[numerical_cols].min())
    return data

# Determine categorical columns based on uniqueness threshold
def get_categorical_columns(data, uniqueness_threshold=0.0005):
    categorical_cols = []
    for col in data.columns:
        unique_ratio = data[col].nunique() / len(data)
        if unique_ratio <= uniqueness_threshold:
            categorical_cols.append(col)
    return categorical_cols

# Encode categorical data using one-hot encoding
def encode_categorical(data, categorical_cols, save_file):
    encoded_data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    encoded_data.to_csv(save_file, index=False)  # Save the encoded data to a CSV file
    return encoded_data

# Build the generator model with L2 regularization
def build_generator_model(latent_dim, num_attributes, l2_factor):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(num_attributes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    return model

# Build the discriminator model with L2 regularization
def build_discriminator_model(num_attributes, l2_factor):
    model = models.Sequential()
    model.add(layers.Dense(128, input_shape=(num_attributes,), kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.Dense(1))
    return model

'''# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)'''

# Wasserstein loss function v2
def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_pred)


# Build the adversarial model
def build_adversarial_model(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Wasserstein GAN class
class TabularWGAN:
    def __init__(self, num_attributes, latent_dim, l2_factor):
        self.num_attributes = num_attributes
        self.latent_dim = latent_dim
        self.l2_factor = l2_factor

        self.generator = self.build_generator_model()
        self.discriminator = self.build_discriminator_model()
        self.adversarial_model = self.build_adversarial_model()


    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        generator_path = os.path.join(directory, "generator.h5")
        discriminator_path = os.path.join(directory, "discriminator.h5")
        adversarial_path = os.path.join(directory, "adversarial.h5")
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
        self.adversarial_model.save(adversarial_path)

    def load_model(self, directory):
        generator_path = os.path.join(directory, "generator.h5")
        discriminator_path = os.path.join(directory, "discriminator.h5")
        adversarial_path = os.path.join(directory, "adversarial.h5")
        self.generator = models.load_model(generator_path)
        self.discriminator = models.load_model(discriminator_path)
        self.adversarial_model = models.load_model(adversarial_path)


    def build_generator_model(self):
        generator = build_generator_model(self.latent_dim, self.num_attributes, self.l2_factor)
        return generator

    def build_discriminator_model(self):
        model = build_discriminator_model(self.num_attributes, self.l2_factor)
        return model

    def build_adversarial_model(self):
        adversarial_model = build_adversarial_model(self.generator, self.discriminator)
        adversarial_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.000005),
                                  loss=wasserstein_loss)
        return adversarial_model

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        generated_data = self.generator.predict(noise)
        return generated_data

    def compute_gradient_penalty(self, real_data, synthetic_data):
        alpha = tf.random.uniform(shape=[real_data.shape[0], 1], minval=0.0, maxval=1.0)
        interpolated_samples = alpha * real_data + (1 - alpha) * synthetic_data

        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            interpolated_predictions = self.discriminator(interpolated_samples)

        gradients = tape.gradient(interpolated_predictions, interpolated_samples)
        gradient_penalty = tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1.0))
        return gradient_penalty

    def train(self, data, batch_size=20000, num_critic=10, convergence_threshold=0.001, average_window=10):
        categorical_cols = get_categorical_columns(data)
        encoded_data = encode_categorical(data, categorical_cols, save_file="encoded_real.csv")

        self.discriminator.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.000005),
                                   loss=wasserstein_loss)

        wasserstein_distance_hist = []
        discriminator_loss_hist = []
        generator_loss_hist = []
        iteration = 0
        wasserstein_distance = float('inf')

        while abs(wasserstein_distance) > convergence_threshold:
            for _ in range(num_critic):
                # Select a random batch of real data
                batch_indices = np.random.randint(0, len(encoded_data), size=batch_size)
                real_data = encoded_data.iloc[batch_indices].values
                real_data = real_data[:, :self.num_attributes]  # Limit the number of attributes

                # Generate a batch of synthetic data
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                synthetic_data = self.generator.predict(noise)

                # Train the discriminator
                with tf.GradientTape() as tape:
                    d_loss_real = tf.reduce_mean(self.discriminator(real_data))
                    d_loss_synthetic = tf.reduce_mean(self.discriminator(synthetic_data))
                    gradient_penalty = self.compute_gradient_penalty(real_data, synthetic_data)
                    d_loss = d_loss_synthetic - d_loss_real + 10.0 * gradient_penalty

                gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
                self.discriminator.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

                discriminator_loss_hist.append(d_loss)


            # Train the generator within the adversarial model
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.adversarial_model.train_on_batch(noise, -np.ones((batch_size, 1)))

            generator_loss_hist.append(g_loss)

            # Calculate Wasserstein distance
            wasserstein_distance = d_loss_real - d_loss_synthetic
            wasserstein_distance_hist.append(wasserstein_distance)

            # Calculate moving average of Wasserstein distance
            avg_wasserstein_distance = np.convolve(wasserstein_distance_hist, np.ones(average_window), 'valid') / average_window

            iteration += 1

            # Print the losses and current Wasserstein distance
            print(f"Iteration: {iteration} [D loss: {d_loss_real:.4f} + {d_loss_synthetic:.4f}] [G loss: {g_loss:.4f}]")
            print(f"Wasserstein Distance: {wasserstein_distance}")

            if iteration == 100 or iteration == 1000 or iteration == 5000:
                # Plot Wasserstein distance, discriminator loss, and generator loss
                fig, axs = plt.subplots(3, 1, figsize=(8, 18))

                # Wasserstein distance
                axs[0].plot(range(len(wasserstein_distance_hist)), wasserstein_distance_hist, label='Wasserstein Distance')
                axs[0].plot(range(average_window - 1, average_window - 1 + len(avg_wasserstein_distance)), avg_wasserstein_distance, label='Moving Average')
                axs[0].set_title('Wasserstein Distance')
                axs[0].set_xlabel('Iteration')
                axs[0].set_ylabel('Wasserstein Distance')
                axs[0].grid(True)

                # Discriminator loss
                avg_discriminator_loss = np.convolve(discriminator_loss_hist, np.ones(average_window), 'valid') / average_window
                axs[1].plot(range(len(discriminator_loss_hist)), discriminator_loss_hist, label='Discriminator Loss')
                axs[1].plot(range(average_window - 1, average_window - 1 + len(avg_discriminator_loss)), avg_discriminator_loss, label='Moving Average')
                axs[1].set_title('Discriminator Loss')
                axs[1].set_xlabel('Iteration')
                axs[1].set_ylabel('Loss')
                axs[1].grid(True)

                # Generator loss
                axs[2].plot(range(len(generator_loss_hist)), generator_loss_hist, label='Generator Loss')
                avg_generator_loss = np.convolve(generator_loss_hist, np.ones(average_window), 'valid') / average_window
                axs[2].plot(range(average_window - 1, average_window - 1 + len(avg_generator_loss)), avg_generator_loss, label='Moving Average')
                axs[2].set_title('Generator Loss')
                axs[2].set_xlabel('Iteration')
                axs[2].set_ylabel('Loss')
                axs[2].grid(True)

                plt.tight_layout()

                # Save the plot
                plot_filename = f'wasserstein_dist_{iteration}.png'
                plt.savefig(plot_filename)
                plt.close()

                # Decode synthetic data
                # To do

                # Export synthetic data
                synthetic_data_df = pd.DataFrame(self.generate_samples(batch_size), columns=data.columns)
                synthetic_data_df.to_csv(f"synth_sample_{iteration}.csv", index=False)


# Main function
def main():
    # Load and preprocess the data
    data = load_data('true_imputed_no_tchol.csv')

    # Normalize the data
    data = normalize_data(data)

    categorical_cols = get_categorical_columns(data)
    encoded_data = encode_categorical(data, categorical_cols, save_file="encoded_real.csv")

    num_attributes = encoded_data.shape[1]
    latent_dim = 500  # input for generator model
    l2_factor = 0.001  # L2 regularization factor

    tabular_gan = TabularWGAN(num_attributes, latent_dim, l2_factor)
    tabular_gan.train(encoded_data)

    # Save the trained model
    #model_directory = '/content/drive/MyDrive/msc dissertation data'
    #tabular_gan.save_model(model_directory)
    tabular_gan.save_model()

if __name__ == '__main__':
    main()
