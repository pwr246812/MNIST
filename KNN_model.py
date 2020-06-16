from mnist_reader import load_mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


X_train, y_train = load_mnist('/Users/marcin/PycharmProjects/untitled4/lab4', kind='train')
X_test, y_test = load_mnist('/Users/marcin/PycharmProjects/untitled4/lab4', kind='t10k')
X_train = [x.reshape(28, 28) for x in X_train]
X_test = [x.reshape(28, 28) for x in X_test]


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def show_images(columns, rows):
    fig = plt.figure(figsize=(10, 10))

    for i in range(1, columns*rows + 1):
        image = Image.fromarray(X_train[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(image)

    plt.show()


def show_one_image(array):
    image = Image.fromarray(array)
    plt.imshow(image)
    plt.show()


def calculate_difference(new_X):
    differences = [0] * 60000

    for i in range(len(X_train)):
        differences[i] = euclides_distance(X_train[i], new_X)

    return differences


def euclides_distance(A, B):
    distance = np.sum((A.astype(int) - B.astype(int)) ** 2)

    return np.sqrt(distance)


def sort_labels(new_X):
    distances = calculate_difference(new_X)
    labels = y_train
    sortedLabels = pd.Series(data=labels, index=distances).sort_index().tolist()

    return sortedLabels


def calculate_label(new_X, k):
    sortedLabels = np.array(sort_labels(new_X)[:k])
    counts = np.bincount(sortedLabels)
    most_common_label = np.argmax(counts)

    return most_common_label


def test_model(k, examples):
    accurate_predictions = 0

    for i in range(len(X_test[:examples])):
        print('Actually checking image:', i + 1)
        prediction = calculate_label(X_test[i], k)
        if prediction == y_test[i]:
            accurate_predictions += 1

    return accurate_predictions/examples*100


def k_selector(examples, k_range):
    best_accuracy = 0
    best_k = 0

    for k in range(1, k_range + 1):
        print('Actually checking k-value:', k)
        actual_accuracy = test_model(k, examples)
        if actual_accuracy >= best_accuracy:
            best_accuracy = actual_accuracy
            best_k = k

    return [best_k, best_accuracy]


### Examples of usage:

### Show images from training dataset.
#show_images(columns=4, rows=5)

### Selecting k-value with best accuracy (examples - how many images from dataset).
#print(k_selector(examples=100, k_range=5))

### Checking the overall model accuracy (examples - how many images from dataset (max: 10000)).
#print('\nCelność modelu to:', str(test_model(k=5, examples=100)) + '%')

### Predicting class for image from test dataset (index - number of image from dataset (max: 10000)).
#index = 293
#print('\nPrediction:', class_names[int(calculate_label(X_test[index], 5))], '\nReal value:', class_names[y_test[293]])
#show_one_image(X_test[index])

