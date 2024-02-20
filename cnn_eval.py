import os
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

random.seed(time())
model = load_model('model.h5')
square_length = 1 # num images is this^2, so 4 would make 16 images
test_dir = 'celeb_src_cnn/test'
celebrities = os.listdir(test_dir)

# func to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize pixel values from [0,255] to [0,1.0)
    return img_array

# function to plot image grid
def plot_image_grid(images, labels, predicted_labels):
    plt.figure(figsize=(square_length * 3, square_length * 3))
    for i in range(min(square_length ** 2, len(images))):
        plt.subplot(square_length, square_length, i + 1)
        plt.imshow(images[i])
        plt.title(f"Actual: {labels[i]}\nPredicted: {predicted_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cnn_predictions.png')

# Load and preprocess the test images
images = []
labels = []
predicted_labels = []
for celebrity in celebrities:
    celebrity_dir = os.path.join(test_dir, celebrity)
    celebrity_images = os.listdir(celebrity_dir)
    for img_name in celebrity_images:
        img_path = os.path.join(celebrity_dir, img_name)
        images.append(img_path)
        labels.append(celebrity)

# Randomly select n x n images
selected_indices = random.sample(range(len(images)), square_length ** 2)
selected_images = [images[i] for i in selected_indices]
selected_labels = [labels[i] for i in selected_indices]

# Preprocess the selected images
preprocessed_images = [preprocess_image(img_path) for img_path in selected_images]

# Predict labels for the selected images
predicted_labels = [celebrities[np.argmax(model.predict(img))] for img in preprocessed_images]

# Load the images for plotting
plot_images = [image.load_img(img_path) for img_path in selected_images]

# Plot a n x n grid
plot_image_grid(plot_images, selected_labels, predicted_labels)
