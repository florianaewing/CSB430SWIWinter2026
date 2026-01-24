import kagglehub
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Download latest version of the dataset
path = kagglehub.dataset_download("datamunge/sign-language-mnist")

# Load CSVs
train_data = pd.read_csv(f"{path}/sign_mnist_train.csv")
test_data = pd.read_csv(f"{path}/sign_mnist_test.csv")

# Separate labels and features
y_train = train_data["label"].values
y_test = test_data["label"].values
X_train = train_data.drop("label", axis=1).values
X_test = test_data.drop("label", axis=1).values

# Normalize and reshape
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# -----------------------
# Simple Dense Model
# -----------------------
dense_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(26, activation="softmax"),
])
dense_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# -----------------------
# Flexible CNN Model Builder
# -----------------------
def build_model(optimizer="adam", use_dropout=False, use_batchnorm=False, l2_reg=0.0):
    model = Sequential()

    # First convolutional block
    model.add(Conv2D(
        32, (3, 3), activation="relu",
        input_shape=(28, 28, 1),
        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
    ))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Second convolutional block
    model.add(Conv2D(
        64, (3, 3), activation="relu",
        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
    ))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Dense layer
    model.add(Dense(
        128, activation="relu",
        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
    ))
    if use_dropout:
        model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(26, activation="softmax"))

    # Optimizer selection
    if optimizer == "adam":
        opt = Adam()
    elif optimizer == "sgd":
        opt = SGD()
    elif optimizer == "rmsprop":
        opt = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -----------------------
# Train CNN Models with Different Optimizers
# -----------------------
optimizers = ["adam", "sgd", "rmsprop"]
histories = {}

for opt in optimizers:
    model = build_model(optimizer=opt, use_dropout=True, use_batchnorm=True, l2_reg=0.001)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        verbose=1
    )
    histories[opt] = history 

histories["adam"].model.save("src/ASL-classifier-model/asl_cnn_model.h5")
# The final model is saved as 'asl_cnn_model.h5'