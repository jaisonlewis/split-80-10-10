import numpy as np
import tensorflow as tf
import os
from glob import glob

#directory locations

image_dir = 'nor_img'
mask_dir = 'norm_img_mask'
label_dir = 'nor_label'

image_files = sorted(glob(os.path.join(image_dir, '*.png')))
mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))
label_files = sorted(glob(os.path.join(label_dir, '*.txt')))

batch_size = 32
shuffle_buffer_size = 1000

# Split data into train, validation, and test sets (80:10:10)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

num_files = len(image_files)
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val

# Generate random indices for splitting
random_indices = np.random.permutation(num_files)
train_idx = random_indices[:num_train]
val_idx = random_indices[num_train:num_train + num_val]
test_idx = random_indices[num_train + num_val:]

# Create arrays of training, validation, and test files (these are filenames)
train_images = np.array(image_files)[train_idx]
val_images = np.array(image_files)[val_idx]
test_images = np.array(image_files)[test_idx]

train_masks = np.array(mask_files)[train_idx]
val_masks = np.array(mask_files)[val_idx]
test_masks = np.array(mask_files)[test_idx]

train_labels = np.array(label_files)[train_idx]
val_labels = np.array(label_files)[val_idx]
test_labels = np.array(label_files)[test_idx]

# Create the save directories for train, validation, and test images, masks, and labels
save_dir = 'saved_data'
os.makedirs(save_dir, exist_ok=True)

train_image_dir = os.path.join(save_dir, 'train_images')
os.makedirs(train_image_dir, exist_ok=True)

val_image_dir = os.path.join(save_dir, 'val_images')
os.makedirs(val_image_dir, exist_ok=True)

test_image_dir = os.path.join(save_dir, 'test_images')
os.makedirs(test_image_dir, exist_ok=True)

train_mask_dir = os.path.join(save_dir, 'train_masks')
os.makedirs(train_mask_dir, exist_ok=True)

val_mask_dir = os.path.join(save_dir, 'val_masks')
os.makedirs(val_mask_dir, exist_ok=True)

test_mask_dir = os.path.join(save_dir, 'test_masks')
os.makedirs(test_mask_dir, exist_ok=True)

train_label_dir = os.path.join(save_dir, 'train_labels')
os.makedirs(train_label_dir, exist_ok=True)

val_label_dir = os.path.join(save_dir, 'val_labels')
os.makedirs(val_label_dir, exist_ok=True)

test_label_dir = os.path.join(save_dir, 'test_labels')
os.makedirs(test_label_dir, exist_ok=True)

# Save training images
for i in range(len(train_images)):
    image = train_images[i]
    filename = os.path.basename(image)
    save_path = os.path.join(train_image_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(image))

# Save validation images
for i in range(len(val_images)):
    image = val_images[i]
    filename = os.path.basename(image)
    save_path = os.path.join(val_image_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(image))

# Save test images
for i in range(len(test_images)):
    image = test_images[i]
    filename = os.path.basename(image)
    save_path = os.path.join(test_image_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(image))

# Save training masks
for i in range(len(train_masks)):
    mask = train_masks[i]
    filename = os.path.basename(mask)
    save_path = os.path.join(train_mask_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(mask))

# Save validation masks
for i in range(len(val_masks)):
    mask = val_masks[i]
    filename = os.path.basename(mask)
    save_path = os.path.join(val_mask_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(mask))

# Save test masks
for i in range(len(test_masks)):
    mask = test_masks[i]
    filename = os.path.basename(mask)
    save_path = os.path.join(test_mask_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(mask))

# Save training labels
for i in range(len(train_labels)):
    label = train_labels[i]
    filename = os.path.basename(label)
    save_path = os.path.join(train_label_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(label))

# Save validation labels
for i in range(len(val_labels)):
    label = val_labels[i]
    filename = os.path.basename(label)
    save_path = os.path.join(val_label_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(label))

# Save test labels
for i in range(len(test_labels)):
    label = test_labels[i]
    filename = os.path.basename(label)
    save_path = os.path.join(test_label_dir, filename)
    tf.io.write_file(save_path, tf.io.read_file(label))