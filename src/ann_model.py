"""Handwritten digit classification on the MNIST dataset using a simple Artificial Neural Network (ANN)."""


import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


def prepare_data(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Flatten, normalize images and one-hot encode labels for ANN training."""
    # Transform the images to 1D vectors of floats
    num_pixels = x_train.shape[1]*x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

    # Normalize the input values
    x_train = x_train / 255
    x_test = x_test / 255

    # Transform the classes labels into a binary matrix
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    num_classes = y_train.shape[1]

    return x_train, y_train, x_test, y_test, num_pixels, num_classes


def ann_model(num_pixels:int, num_classes:int) -> keras.models.Sequential:
    """Build and compile a simple ANN baseline model."""
    # Initialize the model
    model = keras.models.Sequential()

    # Define model input shape
    model.add(keras.layers.Input(shape=(num_pixels,)))

    # Add a hidden dense layer with 8 neurons
    model.add(keras.layers.Dense(units=8,
                                 kernel_initializer='normal',
                                 activation='relu'))
    
    # Add the output dense layer
    model.add(keras.layers.Dense(units=num_classes,
                                 kernel_initializer='normal',
                                 activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def cross_validation(x_train:np.ndarray, y_train:np.ndarray, num_pixels:int, num_classes:int) -> tuple[list, list] :
    """Train and evaluate the model using k-fold cross-validation to assess model stability."""
    k_folds = 5
    histories = []

    # Prepare the cross validation datasets
    k_fold = KFold(n_splits=k_folds, shuffle=True, random_state=1)

    for train_idx, val_idx in k_fold.split(x_train):
        # Select data for train and validation
        x_train_i = x_train[train_idx] 
        y_train_i = y_train[train_idx] 
        x_val_i = x_train[val_idx] 
        y_val_i = y_train[val_idx]

        # Build the model architecture
        model = ann_model(num_pixels, num_classes)

        # Fit the model
        history = model.fit(x_train_i, y_train_i, epochs=5, batch_size=32, validation_data= (x_val_i, y_val_i), verbose=1)

        # Save the training related information in the histories list
        histories.append(history)

    return histories


def display_learning_curves(histories: list) -> None:
    """Plot mean training and validation curves over k-folds."""
    n_epochs = len(histories[0].history["loss"])
    epochs = range(1, n_epochs + 1)

    train_loss = np.array([h.history["loss"] for h in histories])
    val_loss = np.array([h.history["val_loss"] for h in histories])
    train_acc = np.array([h.history["accuracy"] for h in histories])
    val_acc = np.array([h.history["val_accuracy"] for h in histories])

    # Mean and std
    train_loss_mean, train_loss_std = train_loss.mean(axis=0), train_loss.std(axis=0)
    val_loss_mean, val_loss_std = val_loss.mean(axis=0), val_loss.std(axis=0)
    train_acc_mean, train_acc_std = train_acc.mean(axis=0), train_acc.std(axis=0)
    val_acc_mean, val_acc_std = val_acc.mean(axis=0), val_acc.std(axis=0)

    plt.figure(figsize=(10, 8))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.title("Cross Entropy Loss")
    plt.plot(epochs, train_loss_mean, label="Train")
    plt.fill_between(
        epochs,
        train_loss_mean - train_loss_std,
        train_loss_mean + train_loss_std,
        alpha=0.2
    )
    plt.plot(epochs, val_loss_mean, label="Validation")
    plt.fill_between(
        epochs,
        val_loss_mean - val_loss_std,
        val_loss_mean + val_loss_std,
        alpha=0.2
    )
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.title("Classification Accuracy")
    plt.plot(epochs, train_acc_mean, label="Train")
    plt.fill_between(
        epochs,
        train_acc_mean - train_acc_std,
        train_acc_mean + train_acc_std,
        alpha=0.2
    )
    plt.plot(epochs, val_acc_mean, label="Validation")
    plt.fill_between(
        epochs,
        val_acc_mean - val_acc_std,
        val_acc_mean + val_acc_std,
        alpha=0.2
    )
    plt.legend()

    plt.xlabel("Epochs")
    plt.tight_layout()
    plt.show()


def evaluate_on_test(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, num_pixels:int, num_classes: int,) -> float:
    """Train the ANN on the full training set and evaluate accuracy on the test set."""
    # Build the model
    model = ann_model(num_pixels, num_classes)

    # Train on the full training dataset
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

    # Evaluate on the test dataset
    scores = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test accuracy: {scores[1]:.4f}")
    return scores[1]


def main():
    """Run the full ANN training pipeline on the MNIST dataset."""
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test, num_pixels, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    # Train the model and evaluate it throught the cross validation method
    histories = cross_validation(x_train, y_train, num_pixels, num_classes)
    display_learning_curves(histories)

    # Final evaluation on test set
    evaluate_on_test(x_train, y_train, x_test, y_test, num_pixels, num_classes)


if __name__ == "__main__":
    main()