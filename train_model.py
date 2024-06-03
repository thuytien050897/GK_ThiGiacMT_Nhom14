import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize the data
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# One-hot encode the labels
encoder = OneHotEncoder()
y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()  # Convert to dense array
y_test_one_hot = encoder.transform(y_test.reshape(-1, 1)).toarray()  # Convert to dense array

# Define the improved CNN model
inp = Input(shape=(28, 28, 1))

# First Conv Block
cnn = Conv2D(filters=32, kernel_size=3, activation='relu')(inp)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = Dropout(0.2)(cnn)

# Second Conv Block
cnn = Conv2D(filters=64, kernel_size=3, activation='relu')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = Dropout(0.3)(cnn)

# Third Conv Block
cnn = Conv2D(filters=128, kernel_size=3, activation='relu')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = Dropout(0.4)(cnn)

# Fully Connected Layer
f = Flatten()(cnn)
fc1 = Dense(units=128, activation='relu')(f)
fc1 = Dropout(0.5)(fc1)
fc2 = Dense(units=64, activation='relu')(fc1)
fc2 = Dropout(0.5)(fc2)
out = Dense(units=10, activation='softmax')(fc2)

model = Model(inputs=inp, outputs=out)
model.summary()

# Compile the model
optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer1, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train_one_hot, batch_size=64, epochs=70, validation_data=(X_test_scaled, y_test_one_hot))

# Save the model
model.save("mnist_cnn.h5")

