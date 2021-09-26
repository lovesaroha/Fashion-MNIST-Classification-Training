# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model on  Fashion MNIST dataset.
from tensorflow import keras
import matplotlib.pyplot as plt

# Parameters.
learningRate = 0.01
epochs = 10
batchSize = 256

# Load fashion MNIST dataset.
mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (validation_images,
                                     validation_labels) = mnist.load_data()

# Normalize inputs.
training_images = training_images / 255
validation_images = validation_images / 255

# Create model with 10 output units for classification.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Set loss function and optimizer.
model.compile(optimizer=keras.optimizers.Adam(learningRate),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.95:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
history = model.fit(training_images, training_labels, batch_size=batchSize, epochs=epochs, callbacks=[
                    checkAccuracy], validation_data=(validation_images, validation_labels))

# Predict on a random image.
image = validation_images[3]
prediction = model.predict(image.reshape(1, 28, 28))

# Show image.
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
plt.show()

# Labels.
labels = {0: "T-Shirt/Top" , 1: "Trouser" , 2: "Pullover" , 3: "Dress" , 4: "Coat" , 5: "Sandal" , 6: "Shirt" , 7: "Sneaker" , 8: "Bag" , 9: "Ankle Boot"}

label = 0
# Show label.
for i in range(10):
  if prediction[0][label] < prediction[0][i]:
    label = i

print("It is a :" , labels[label])    