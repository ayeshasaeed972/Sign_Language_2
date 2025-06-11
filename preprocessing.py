import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

# Parameters
DATASET_DIR = "dataset"  # folder with class subfolders
IMG_SIZE = 64  # image size (64x64)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Prepare lists for images and labels
images = []
labels = []

# Get class names (folder names in dataset dir)
class_names = sorted(os.listdir(DATASET_DIR))
print(f"Classes found: {class_names}")

# Load images and labels
for idx, class_name in enumerate(class_names):
    class_folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_folder):
        continue

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        # Load image in grayscale or color
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(idx)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize images (0 to 1)
images = images / 255.0

# Reshape images for CNN (samples, height, width, channels)
images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# One-hot encode labels
labels = to_categorical(labels, num_classes=len(class_names))

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)

# Print summary
print("=== Preprocessing Summary ===")
print(f"Total samples: {len(images)}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]}")
print(f"Classes: {class_names}")

# Save to utils folder
os.makedirs("utils", exist_ok=True)
np.savez("utils/preprocessing_data.npz",
         X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, classes=class_names)

print("✅ Preprocessing complete. Data saved to utils/preprocessing_data.npz")

