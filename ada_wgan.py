import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import matplotlib.pyplot as plt
import os
import re

# Load and preprocess the input data
def load_data(filename):
    data = pd.read_csv(filename)
    return data


# Normalize numerical data using min-max scaling
def normalize_data(data):
    numerical_cols = data.select_dtypes(include=np.number).columns
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].min()) / (
            data[numerical_cols].max() - data[numerical_cols].min())
    return data


# Determine categorical columns based on uniqueness threshold
def get_categorical_columns(data, uniqueness_threshold=0.0005):
    categorical_cols = []
    for col in data.columns:
        unique_ratio = data[col].nunique() / len(data)
        if unique_ratio <= uniqueness_threshold:
            categorical_cols.append(col)
    return categorical_cols


# Encode original data
def encode_categorical(data, categorical_cols, save_file=None):
    encoded_data = pd.get_dummies(data, columns=categorical_cols, prefix_sep='_')
    if save_file:
        encoded_data.to_csv(save_file, index=False)
    return encoded_data


def decode_categorical(encoded_data, categorical_cols):
    decoded_data = pd.DataFrame()

    for col in encoded_data.columns:
        if col in categorical_cols:
            separator_index = col.rfind("_")
            prefix = col[:separator_index]
            matching_cols = [c for c in encoded_data.columns if c.startswith(prefix)]

            if not matching_cols:
                print(f"Warning: Encoded columns not found for {col}.")
                decoded_data[col] = np.nan
            else:
                categorical_value = encoded_data[matching_cols].idxmax(axis=1).apply(lambda x: x.split("_")[-1])
                decoded_data[prefix] = categorical_value
        else:
            decoded_data[col] = encoded_data[col]  # Add non-categorical columns

    return decoded_data


# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_pred)

# Build the discriminator model with L2 regularization and RMSprop optimizer
def build_discriminator_model(num_attributes, l2_factor):
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(num_attributes,), kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1))
    return model


# Build the generator model with L2 regularization and Adamax optimizer
def build_generator_model(latent_dim, num_attributes, l2_factor):
    model = models.Sequential()
    model.add(layers.Dense(512, input_dim=latent_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(num_attributes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    return model


# Build the adversarial model
def build_adversarial_model(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


#FULLY ADAPTIVE VERSION
# Wasserstein GAN class
class TabularWGAN:
    def __init__(self, num_attributes, latent_dim, l2_factor, learning_rate, clipvalue):
        self.num_attributes = num_attributes
        self.latent_dim = latent_dim
        self.l2_factor = l2_factor
        self.learning_rate = learning_rate
        self.clipvalue = clipvalue

        self.generator = build_generator_model(latent_dim, num_attributes, l2_factor)
        self.discriminator = build_discriminator_model(num_attributes, l2_factor)
        self.adversarial_model = self.build_adversarial_model()

    def build_adversarial_model(self):
        adversarial_model = build_adversarial_model(self.generator, self.discriminator)
        adversarial_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                                  loss=wasserstein_loss)
        return adversarial_model

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        noise_with_noise = noise + np.random.normal(0, 0.1, noise.shape)  # Add noise to the input
        generated_data = self.generator.predict(noise_with_noise)
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

    def train(self, data, batch_size=20000, num_critic=10, convergence_threshold=0.0001, average_window=10):
        categorical_cols = get_categorical_columns(data)
        encoded_data = encode_categorical(data, categorical_cols, save_file="encoded_real.csv")

        self.discriminator.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, clipvalue=self.clipvalue),
                                   loss=wasserstein_loss)

        wasserstein_distance_hist = []
        discriminator_loss_hist = []
        generator_loss_hist = []
        iteration = 0
        wasserstein_distance = float('inf')

        # Define the learning rate schedule
        lr_schedule = PiecewiseConstantDecay(
            boundaries=[500, 1000, 5000],
            values=[self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01, self.learning_rate * 0.001]
        )

        while abs(wasserstein_distance) > convergence_threshold:
            # Update the learning rate for the optimizers
            self.adversarial_model.optimizer.lr.assign(lr_schedule(iteration))
            self.discriminator.optimizer.lr.assign(lr_schedule(iteration))

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

            # Apply adaptive gradient clipping
            for variable in self.discriminator.trainable_variables:
                variable.assign(tf.clip_by_value(variable, -self.clipvalue, self.clipvalue))

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
                plot_filename = f'/content/drive/MyDrive/wasserstein_dist_{iteration}.png'
                plt.savefig(plot_filename)
                plt.close()

        # Import real data
        real_data = load_data('/content/drive/MyDrive/true_imputed_no_tchol.csv')
        real_data = real_data.iloc[:, 1:]

        # Categorical columns in encoded data
        categorical_cols = get_categorical_columns(real_data)

        # Decode synthetic data
        synth = pd.DataFrame(self.generate_samples(200000), columns=data.columns)
        synth_dec = decode_categorical(synth, categorical_cols)

        # Denormalize decoded synthetic data
        synth_denorm = denormalize_data(synth_dec, real_data)
        synth_denorm.to_csv('/content/drive/MyDrive/synthetic data.csv', index=False)

        self.save_model("/content/drive/MyDrive/")


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
        plot_filename = f'/content/drive/MyDrive/wasserstein_dist_{iteration}.png'
        plt.savefig(plot_filename)
        plt.close()


# Main function
def main():
    # Check if the model is already present in the directory
    model_directory = "/content/drive/MyDrive/"
    model_present = os.path.exists(os.path.join(model_directory, "generator.h5"))

    if model_present:
        # Load the model
        tabular_wgan = TabularWGAN(0, 0, 0, 0)  # Provide a dummy value for learning_rate
        tabular_wgan.load_model(model_directory)

        # Generate samples using the trained model
        num_samples = 200000
        generated_data = tabular_wgan.generate_samples(num_samples)
        generated_data = pd.DataFrame(generated_data, columns=data.columns)

        # Save generated samples to CSV
        generated_data.to_csv('/content/drive/MyDrive/generated_samples.csv', index=False)

    else:
        # Load and preprocess the data
        data = load_data('/content/drive/MyDrive/true_imputed_no_tchol.csv')

        # Remove the first column
        data = data.iloc[:, 1:]

        categorical_cols = get_categorical_columns(data)
        encoded_data = encode_categorical(data, categorical_cols, save_file='/content/drive/MyDrive/encoded_real.csv')

        # Normalize the data
        data = normalize_data(encoded_data)
        data.to_csv(f"norm_real.csv", index=False)

        num_attributes = encoded_data.shape[1]
        latent_dim = 500  # input for generator model
        l2_factor = 0.001  # L2 regularization factor
        learning_rate = 0.00005  # Initial learning rate

        clipvalue = 0.001

        tabular_wgan = TabularWGAN(num_attributes, latent_dim, l2_factor, learning_rate, clipvalue)
        tabular_wgan.train(data)

        # Generate samples using the trained model
        num_samples = 200000
        generated_data = tabular_wgan.generate_samples(num_samples)
        generated_data = pd.DataFrame(generated_data, columns=data.columns)

        # Save generated samples to CSV
        generated_data.to_csv('/content/drive/MyDrive/generated_samples.csv', index=False)


if __name__ == '__main__':
    main()

