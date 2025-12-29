import numpy as np
import pytest

from ann_model import ann_model, cross_validation, prepare_data


@pytest.fixture
def small_mnist_dataset():
    x_train = np.random.randint(0, 255, size=(10, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, size=(10,))
    x_test = np.random.randint(0, 255, size=(4, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, size=(4,))
    return x_train, y_train, x_test, y_test


def test_prepare_data_shapes(small_mnist_dataset):
    x_train, y_train, x_test, y_test = small_mnist_dataset
    x_train, y_train, x_test, y_test, num_pixels, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    # Check shapes
    assert x_train.shape == (10, 28 * 28)
    assert x_test.shape == (4, 28 * 28)
    assert y_train.shape == (10, num_classes)
    assert y_test.shape == (4, num_classes)

    # Check normalization
    assert np.all((x_train >= 0) and (x_train <= 1))
    assert np.all((x_test >= 0) and (x_test <= 1))


def test_ann_model_prediction():
    model = ann_model(num_pixels=784, num_classes=10)

    x_sample = np.random.rand(1, 784).astype("float32")
    prediction = model.predict(x_sample, verbose=0)

    assert prediction.shape == (1, 10)
    assert np.isclose(np.sum(prediction), 1.0, atol=1e-5)


def test_cross_validation_runs(small_mnist_dataset):
    x_train, y_train, x_test, y_test = small_mnist_dataset
    x_train, y_train, x_test, y_test, num_pixels, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    model = ann_model(num_pixels, num_classes)
    histories, accuracy_scores = cross_validation(x_train, y_train, x_test, y_test, model)

    assert len(histories) > 0
    assert len(accuracy_scores) > 0
    assert all(0 <= accuracy <= 1 for accuracy in accuracy_scores)