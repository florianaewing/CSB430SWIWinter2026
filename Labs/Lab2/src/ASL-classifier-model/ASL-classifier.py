import kagglehub
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import time
from functools import wraps
import os

# -----------------------
# Decorator to log training
def log_training(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]  # First arg is always the instance
        print(f"\nStarting training for {self.model_name}...")
        start_time = time.time()

        history = func(*args, **kwargs)

        end_time = time.time()
        duration = end_time - start_time
        best_val_acc = self.best_val_acc
        print(f"Finished training {self.model_name} in {duration:.2f}s")
        print(f"Best validation accuracy for {self.model_name}: {best_val_acc:.4f}\n")
        return history
    return wrapper

# -----------------------
# ASL Model Class
class ASLModel:
    def __init__(self, model_name="Model", input_shape=(28,28,1), num_classes=26):
        self.model = None
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.history = None
        self.early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    # -----------------------
    def build_dense(self):
        self.model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.num_classes, activation="softmax")
        ])
        self.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    # -----------------------
    def build_cnn(self, conv_layers=None, dense_layers=None, optimizer="adam",
                  use_dropout=False, use_batchnorm=False, l2_reg=0.0):
        """
        Flexible CNN builder.
        conv_layers: list of tuples (filters, kernel_size, pool_size)
        dense_layers: list of integers (units)
        """
        self.model = Sequential()
        
        if conv_layers is None:
            conv_layers = [(16, (3,3), (2,2)), (32, (3,3), (2,2))]
        if dense_layers is None:
            dense_layers = [64]

        # Add convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(conv_layers):
            if i == 0:
                self.model.add(Conv2D(filters, kernel_size, activation="relu",
                                      input_shape=self.input_shape,
                                      kernel_regularizer=l2(l2_reg) if l2_reg>0 else None))
            else:
                self.model.add(Conv2D(filters, kernel_size, activation="relu",
                                      kernel_regularizer=l2(l2_reg) if l2_reg>0 else None))
            if use_batchnorm:
                self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size))

        self.model.add(Flatten())

        # Add dense layers
        for units in dense_layers:
            self.model.add(Dense(units, activation="relu",
                                 kernel_regularizer=l2(l2_reg) if l2_reg>0 else None))
            if use_dropout:
                self.model.add(Dropout(0.6))

        # Output layer
        self.model.add(Dense(self.num_classes, activation="softmax"))

        # Optimizer selection
        if optimizer=="adam":
            opt = Adam()
        elif optimizer=="sgd":
            opt = SGD()
        elif optimizer=="rmsprop":
            opt = RMSprop()
        else:
            raise ValueError("Unsupported optimizer")

        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    # -----------------------
    @log_training
    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[self.early_stop]
        )
        return self.history

    # -----------------------
    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, acc

    # Lambda property for best val accuracy
    best_val_acc = property(lambda self: max(self.history.history["val_accuracy"])
                            if self.history and self.history.history["val_accuracy"] else None)

    # -----------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

# -----------------------
# Data Loading and Preprocessing
path = kagglehub.dataset_download("datamunge/sign-language-mnist")
train_data = pd.read_csv(f"{path}/sign_mnist_train.csv")
test_data = pd.read_csv(f"{path}/sign_mnist_test.csv")

y_train = train_data["label"].values
y_test = test_data["label"].values
X_train = train_data.drop("label", axis=1).values
X_test = test_data.drop("label", axis=1).values

X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# -----------------------
# Train Dense Model
dense_model = ASLModel("Dense Model")
dense_model.build_dense()
dense_model.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=64)
loss, acc = dense_model.evaluate(X_test, y_test)
print(f"Dense model final evaluation - loss: {loss:.4f}, accuracy: {acc:.4f}")

# -----------------------
# Train CNN Models with flexible architecture
optimizers = ["adam", "sgd", "rmsprop"]
cnn_models = {}
for opt in optimizers:
    cnn = ASLModel(f"CNN ({opt})")
    cnn.build_cnn(
        conv_layers=[(16,(3,3),(2,2)), (32,(3,3),(2,2))],  # Can customize later
        dense_layers=[64],
        optimizer=opt,
        use_dropout=True,
        use_batchnorm=True,
        l2_reg=0.005
    )
    cnn.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=64)
    cnn_models[opt] = cnn

# -----------------------
# Automatic model comparison table
import pandas as pd

comparison = []
# Dense model
comparison.append({
    "Model": dense_model.model_name,
    "Best Val Accuracy": dense_model.best_val_acc,
    "Test Accuracy": dense_model.evaluate(X_test, y_test)[1]
})

# CNN models
for cnn in cnn_models.values():
    comparison.append({
        "Model": cnn.model_name,
        "Best Val Accuracy": cnn.best_val_acc,
        "Test Accuracy": cnn.evaluate(X_test, y_test)[1]
    })

df_comparison = pd.DataFrame(comparison).sort_values(by="Best Val Accuracy", ascending=False)
print("\nModel Comparison Table:")
print(df_comparison)

# -----------------------
# Save models
dense_model.save("src/ASL-classifier-model/asl_dense_model.h5")
cnn_models["adam"].save("src/ASL-classifier-model/asl_cnn_model.h5")
