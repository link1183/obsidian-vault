import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

# --- Configuration ---
NUM_CLASSES = 10
MODEL_SAVE_PATH = "cifar10_cnn_model.keras"

# 1. Load the CIFAR-10 Dataset
print("\n--- Loading CIFAR-10 Dataset ---")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Dataset characteristics
input_shape = (32, 32, 3)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"{x_train.shape[0]} train samples")
print(f"{x_test.shape[0]} test samples")
print(f"Number of classes: {NUM_CLASSES}")

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 2. Preprocess the Data
print("\n--- Preprocessing Data ---")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print("Pixel values scaled to [0, 1]")

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
print("Labels converted to one-hot encoding")
print(f"y_train shape (after one-hot): {y_train.shape}")  # (50000, 10)
print(f"y_test shape (after one-hot): {y_test.shape}")  # (10000, 10)


# 3. Build the CNN Model
print("\n--- Building the CNN Model ---")
# Same simple architecture
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        # Final Dense layer automatically adapts to NUM_CLASSES = 10
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.summary()

# 4. Compile the Model
print("\n--- Compiling the Model ---")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 5. Train the Model
print("\n--- Training the Model ---")
batch_size = 128
epochs = 15

history = model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)
print("--- Training Complete ---")

# 6. Evaluate the Model
print("\n--- Evaluating Model Performance ---")
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# --- Save the Trained Model ---
print(f"\n--- Saving the trained model to {MODEL_SAVE_PATH} ---")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

# 7. Visualize Training History
print("\n--- Visualizing Training History ---")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy (CIFAR-10)")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss (CIFAR-10)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")

plt.tight_layout()
plt.show()

print("\n--- Training script finished ---")
