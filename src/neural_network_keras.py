# Import necessary libraries
import numpy as np  # Library for mathematical operations and numerical array manipulation
from keras.models import Sequential # Import the Sequential model from Keras
from keras.layers import Dense, Input  # Layers to build the neural network
from keras.utils import to_categorical  # Utility to convert labels to one-hot encoding format
from keras.datasets import mnist  # MNIST dataset of handwritten digits
import matplotlib.pyplot as plt  # Library for generating plots

# Define the main function to train and evaluate the model
def train_and_evaluate():
    """
    This function loads the MNIST data, trains a neural network, and evaluates its performance.
    """

    # Load the training and test data from MNIST
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Print information about the loaded data
    print("Shape of training data:", train_data_x.shape)
    print("Label of the first training example:", train_labels_y[1])
    print("Shape of test data:", test_data_x.shape)
    
    # Visualize an example training image
    plt.imshow(train_data_x[1], cmap="gray")
    plt.title("Example Training Image")
    plt.show()

    # Data normalization:
    # Convert images into 28x28 vectors (a single array of 784 values)
    # and normalize by dividing by 255 (to scale the values to the range [0,1])
    x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
    y_train = to_categorical(train_labels_y)  # Convert labels to one-hot format

    x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255
    y_test = to_categorical(test_labels_y)  # Convert test labels to one-hot format

    # Define the architecture of the neural network
    model = Sequential([
        Input(shape=(28*28,)),  # Input layer with 784 neurons (image size)
        Dense(512, activation='relu'),  # Hidden layer with 512 neurons and ReLU activation
        Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit 0-9), softmax activation
    ])

    # Compile the model
    model.compile(
        optimizer='rmsprop',  # Optimizer to adjust weights (RMSprop works well for deep networks)
        loss='categorical_crossentropy',  # Loss function for multiclass classification
        metrics=['accuracy']  # Evaluation metric (accuracy)
    )

    # Train the model
    model.fit(x_train, y_train, epochs=8, batch_size=128)  # 8 epochs, mini-batch of 128 images

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)  # Calculate loss and accuracy on test data
    print(f"Test accuracy: {accuracy:.4f}")  # Display the final accuracy

    print("Training and evaluation completed successfully.")  # Completion message

# Avoid automatic execution if this script is imported into another module
if __name__ == "__main__":
    train_and_evaluate()  # Call the main function if the script is run directly
