"""Handwritten digit classification on the MNIST dataset using a simple Convolutionnal Neural Network (CNN)."""


import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


def prepare_data(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], int]:
    """Reshape, normalize images and one-hot encode labels for CNN training."""
    # Reshape the data to be of size [samples][width][height][channels]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')
    input_shape = (x_train.shape[1], x_train.shape[2], 1)

    # Normalize the input values
    x_train = x_train / 255
    x_test = x_test / 255

    # Transform the classes labels into a binary matrix
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    num_classes = y_train.shape[1]

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def cnn_model(input_shape:tuple[int, int, int], num_classes:int) -> keras.models.Sequential:
    """Build and compile a simple CNN model."""
    # Initialize the model
    model = keras.models.Sequential()

    # Define model input shape
    model.add(keras.layers.Input(shape=input_shape))

    # Add a convolutionnal layer with 30 filters of size 5x5
    model.add(keras.layers.Conv2D(filters=30,
                                 kernel_size=(5,5),
                                 activation='relu'))
    
    # Add a MaxPooling layer
    model.add(keras.layers.MaxPool2D())

    # Add a convolutionnal layer with 15 filters of size 3x3
    model.add(keras.layers.Conv2D(filters=15,
                                  kernel_size=(3,3),
                                  activation='relu'))
    
    # Add a MaxPooling layer
    model.add(keras.layers.MaxPool2D())

    # Add a regularization layer
    model.add(keras.layers.Dropout(rate=0.2))

    # Add a flattening layer
    model.add(keras.layers.Flatten())
    
    # Add two hidden dense layers
    model.add(keras.layers.Dense(units=128,
                                 activation='relu'))
    
    model.add(keras.layers.Dense(units=50,
                                 activation='relu'))
    
    # Add the output dense layer
    model.add(keras.layers.Dense(units=num_classes,
                                 activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def cross_validation(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, input_shape:tuple[int, int, int], num_classes:int) -> tuple[list, list] :
    """Train and evaluate the model using k-fold cross-validation."""
    k_folds = 5
    histories, accuracy_scores = [], []

    # Prepare the cross validation datasets
    k_fold = KFold(n_splits=k_folds, shuffle=True, random_state=1)

    for train_idx, val_idx in k_fold.split(x_train):
        # Select data for train and validation
        x_train_i = x_train[train_idx] 
        y_train_i = y_train[train_idx] 
        x_val_i = x_train[val_idx] 
        y_val_i = y_train[val_idx]

        # Build the model architecture
        model = cnn_model(input_shape, num_classes)

        # Fit the model
        history = model.fit(x_train_i, y_train_i, epochs=5, batch_size=32, validation_data= (x_val_i, y_val_i), verbose=1)

        # Save the training related information in the histories list
        histories.append(history)

        # Evaluate the model on the test dataset
        scores = model.evaluate(x_test, y_test, verbose=0)

        # Save the accuracy in the accuracyScores list
        accuracy_scores.append(scores[1])

    return histories, accuracy_scores


def display_learning_curves(histories: list, accuracyScores: list) -> None:
    """Display loss and accuracy curves for each cross-validation fold."""
    for i in range(len(histories)):
        # Plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='green', label='train')
        plt.plot(histories[i].history['val_loss'], color='red', label='test')

        # PLot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='green', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='red', label='test')

    plt.show()


def main():
    """Run the full CNN training pipeline on the MNIST dataset."""
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    # Train the model and evaluate it throught the cross validation method
    histories, accuracy_scores = cross_validation(x_train, y_train, x_test, y_test, input_shape, num_classes)

    # Display system performance
    display_learning_curves(histories, accuracy_scores)


if __name__ == "__main__":
    main()