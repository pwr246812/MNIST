import tensorflow as tf
from tensorflow import keras
from mnist_reader import load_mnist
import numpy as np
import matplotlib.pyplot as plt
import random


X_test, y_test = load_mnist('/Users/marcin/PycharmProjects/untitled4/lab4', kind='t10k')
X_train, y_train = load_mnist('/Users/marcin/PycharmProjects/untitled4/lab4', kind='train')
X_train = [x.reshape(28, 28, 1) for x in X_train]
X_test = [x.reshape(28, 28, 1) for x in X_test]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_train = np.divide(X_train, 255)
X_test = np.divide(X_test, 255)

model = keras.Sequential()
model.add(keras.layers.Conv2D(28, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(len(class_names), activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('\nModel training ...\n')
model.fit(X_train, y_train, epochs=10)
print('\nTesting model on test data ...\n')
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print('\nTest accuracy: ', test_acc, '(~' + str(round(test_acc, 2)*100) + '%)')

index = random.randint(0, len(X_test))
test_image = X_test[index]
test_image = (np.expand_dims(test_image, 0))
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
single_prediction = probability_model.predict(test_image)

plt.grid(False)
plt.xticks(range(len(class_names)), class_names, rotation=45)
plot = plt.bar(range(10), single_prediction[0], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(single_prediction[0])
plot[predicted_label].set_color('red')
plot[y_test[index]].set_color('green')
plt.xlabel('Prediction')
plt.title('Real value: ' + str(class_names[y_test[index]]))
plt.show()
