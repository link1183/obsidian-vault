import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys

print(f"TensorFlow version: {tf.__version__}")

# --- Configuration ---
MODEL_PATH = "cifar10_cnn_model.keras"
INPUT_SHAPE = (32, 32)
CLASS_NAMES = [
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
NUM_CLASSES = len(CLASS_NAMES)


# --- Function to Load and Preprocess a Single Image ---
def load_and_preprocess_image(img_path, target_size):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array_scaled = img_array / 255.0
        img_batch = np.expand_dims(img_array_scaled, axis=0)
        return img_batch
    except FileNotFoundError:
        print(f"Error: Image file not found at '{img_path}'")
        return None
    except Exception as e:
        print(f"Error processing image '{img_path}': {e}")
        return None


# --- Load the Trained Model ---
print(f"\nLoading trained CIFAR-10 model from {MODEL_PATH}...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure you have run the training script and saved '{MODEL_PATH}'.")
    sys.exit(1)

# --- Get Image Path from Command Line Argument ---
if len(sys.argv) < 2:
    print("\nUsage: python predict_image_cifar10.py <path_to_your_image>")
    image_path_to_predict = ""
else:
    image_path_to_predict = sys.argv[1]

# --- Preprocess the Input Image ---
print(f"\nPreprocessing image: {image_path_to_predict}")
preprocessed_img = load_and_preprocess_image(image_path_to_predict, INPUT_SHAPE)

if preprocessed_img is None:
    sys.exit(1)

# --- Make Prediction ---
print("\nPredicting image class...")
predictions = model.predict(preprocessed_img)
# 'predictions' is a numpy array of shape (1, 10) containing raw probabilities (softmax output)

# --- Interpret the Prediction ---
# The raw prediction array for the first (and only) image in the batch
class_probabilities = predictions[0]

# Find the index and confidence of the highest probability class
predicted_index = np.argmax(class_probabilities)
top_confidence = class_probabilities[predicted_index] * 100

if 0 <= predicted_index < NUM_CLASSES:
    predicted_name = CLASS_NAMES[predicted_index]
else:
    predicted_name = "Unknown Index"

print(f"\n--- Top Prediction ---")
print(f"  Predicted Class: '{predicted_name}' (Index: {predicted_index})")
print(f"  Confidence:       {top_confidence:.2f}%")

# --- <<< NEW: Display Confidence for ALL Classes >>> ---
print("\n--- Confidence Scores for All Classes ---")
# Sort indices by probability descending to show highest first (optional but nice)
sorted_indices = np.argsort(class_probabilities)[
    ::-1
]  # Get indices from highest to lowest prob
for i in sorted_indices:
    class_name = CLASS_NAMES[i]
    confidence = class_probabilities[i] * 100
    print(f"  {i:2d}: {class_name:<12} {confidence:>6.2f}%")

# Or simply iterate 0-9:
# for i in range(NUM_CLASSES):
#     class_name = CLASS_NAMES[i]
#     confidence = class_probabilities[i] * 100
#     # Add an indicator for the predicted class
#     indicator = " <<< Predicted" if i == predicted_index else ""
#     print(f"  Class {i:2d} ({class_name:<10}): {confidence:>6.2f}% {indicator}")
# --- End of New Section ---

# --- Optional: Display the image with the prediction ---
try:
    img_display = plt.imread(image_path_to_predict)
    plt.imshow(img_display)
    plt.title(f"Input Image\nPredicted: '{predicted_name}' ({top_confidence:.1f}%)")
    plt.axis("off")
    plt.show()
except Exception as e:
    print(f"\nWarning: Could not display the image using matplotlib. {e}")

print("\n--- Inference script finished ---")
