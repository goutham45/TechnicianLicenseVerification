import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

# Load dataset
data = tf.keras.utils.image_dataset_from_directory(
    'data', 
    image_size=(256, 256), 
    batch_size=32
)

# Get class names
class_names = data.class_names
num_classes = len(class_names)

data = data.map(lambda x, y: (x / 255.0, tf.one_hot(y, num_classes)))

dataset_size = len(list(data))
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.2)
test_size = dataset_size - (train_size + val_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size)

# Define model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train model
model.fit(train, epochs=15, validation_data=val)

# Save model
model.save("model_weights.h5")

print("Model training complete. Weights saved as 'model_weights.h5'")
