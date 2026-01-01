import numpy as np
import pytest

from cnn_model import cnn_model, cross_validation, evaluate_on_test, prepare_data


@pytest.fixture
def small_mnist_dataset():
    x_train = np.random.randint(0, 255, size=(30, 28, 28), dtype=np.uint8)
    y_train = np.tile(np.arange(10), 3)

    x_test = np.random.randint(0, 255, size=(10, 28, 28), dtype=np.uint8)
    y_test = np.arange(10)

    return x_train, y_train, x_test, y_test


def test_prepare_data_shapes(small_mnist_dataset):
    x_train, y_train, x_test, y_test = small_mnist_dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    assert x_train.shape == (30, 28, 28, 1)
    assert x_test.shape == (10, 28, 28, 1)
    assert y_train.shape == (30, 10)
    assert y_test.shape == (10, 10)

    assert np.all((x_train >= 0) & (x_train <= 1))
    assert np.all((x_test >= 0) & (x_test <= 1))


def test_cnn_model_prediction():
    model = cnn_model(input_shape=(28, 28, 1), num_classes=10)

    x_sample = np.random.rand(1, 28, 28, 1).astype("float32")
    prediction = model.predict(x_sample, verbose=0)

    assert prediction.shape == (1, 10)
    assert np.isclose(np.sum(prediction), 1.0, atol=1e-5)


def test_cross_validation_runs(small_mnist_dataset):
    x_train, y_train, x_test, y_test = small_mnist_dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    histories = cross_validation(x_train, y_train, input_shape, num_classes)

    assert len(histories) > 0


def test_evaluation_on_test(small_mnist_dataset):
    x_train, y_train, x_test, y_test = small_mnist_dataset
    x_train, y_train, x_test, y_test, num_pixels, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    accuracy = evaluate_on_test(x_train, y_train, x_test, y_test, num_pixels, num_classes)

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1