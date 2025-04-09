import tensorflow as tf
from tensorflow import keras
from PIL import Image  # Import Pillow
import os
import sys

print(f"TensorFlow version: {tf.__version__}")

# --- Configuration ---
OUTPUT_DIR = "cifar10_test_images"  # Directory to save the images
NUM_IMAGES_TO_SAVE = 1000  # How many test images to save? (Max: 10000)

# --- CIFAR-10 Class Names (for naming files) ---
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

# --- Load CIFAR-10 Dataset ---
print("\nLoading CIFAR-10 dataset...")
# We only need the test set for this script
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(f"Loaded {len(x_test)} test images.")

# --- Create Output Directory ---
if not os.path.exists(OUTPUT_DIR):
    print(f"Creating directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR)
else:
    print(f"Output directory already exists: {OUTPUT_DIR}")

# --- Limit the Number of Images ---
if NUM_IMAGES_TO_SAVE > len(x_test):
    print(
        f"Warning: Requested to save {NUM_IMAGES_TO_SAVE} images, but test set only has {len(x_test)}. Saving all test images."
    )
    NUM_IMAGES_TO_SAVE = len(x_test)
elif NUM_IMAGES_TO_SAVE <= 0:
    print("Warning: NUM_IMAGES_TO_SAVE set to 0 or less. No images will be saved.")
    sys.exit(0)  # Exit cleanly
else:
    print(f"Preparing to save the first {NUM_IMAGES_TO_SAVE} test images...")


# --- Save Images ---
print("Saving images...")
saved_count = 0
for i in range(NUM_IMAGES_TO_SAVE):
    # Get the image data (NumPy array, shape 32x32x3, dtype uint8)
    img_array = x_test[i]

    # Get the true class label index (y_test is shape [10000, 1])
    label_index = y_test[i][0]
    class_name = CLASS_NAMES[label_index]

    # Create the filename (e.g., cifar10_test_00005_cat.png)
    # Using leading zeros for better sorting
    filename = f"cifar10_test_{i:05d}_{class_name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        # Convert the NumPy array to a Pillow Image object
        # Mode 'RGB' is appropriate for a 3-channel color image
        img = Image.fromarray(img_array, "RGB")

        # Save the image
        img.save(filepath)
        saved_count += 1

        # Print progress occasionally
        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{NUM_IMAGES_TO_SAVE} images...")

    except Exception as e:
        print(f"Error saving image {i} ({filename}): {e}")


print(f"\nFinished saving {saved_count} images to the '{OUTPUT_DIR}' directory.")
