import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential

"""## Prepare Dataset"""

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255.0

print("TRAIN IMAGES: ", train_images.shape)
print("TEST IMAGES: ", test_images.shape)

"""## Create Model"""

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Changed to softmax
])

"""## Compile Model"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Changed
              metrics=['accuracy'])

model.summary()

"""## Train Model"""

epochs = 10
history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

"""## Visualize Training Results"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy and Loss')
plt.show()

"""## Test Image"""

image = train_images[1].reshape(1, 28, 28, 1)
model_pred = np.argmax(model.predict(image), axis=-1)
plt.imshow(image.reshape(28, 28), cmap='gray')
print(f'Prediction of model: {model_pred[0]}')

"""## Test Multiple Images"""

images = test_images[3:7]
for i, test_image in enumerate(images, start=1):
    test_image_reshaped = test_image.reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(test_image_reshaped), axis=-1)

    plt.subplot(2, 2, i)
    plt.axis('off')
    plt.title(f"Predicted digit: {prediction[0]}")
    plt.imshow(test_image.squeeze(), cmap='gray')

plt.show()

"""## Save and Load Model"""

model.save("tf-cnn-model.h5")
loaded_model = models.load_model("tf-cnn-model.h5")

image = train_images[2].reshape(1, 28, 28, 1)
model_pred = np.argmax(loaded_model.predict(image), axis=-1)
plt.imshow(image.reshape(28, 28), cmap='gray')
print(f'Prediction of model: {model_pred[0]}')
