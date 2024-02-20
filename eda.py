# This file is for exploratory data analysis

import os
from PIL import Image

data_dir = 'celeb_src'

total_images = 0
avg_images_per_celebrity = 0
min_images_per_celebrity = 1800 # Total number of images set to be 'minimum' originally
max_images_per_celebrity = 0
total_image_size = 0

num_square_images = 0
num_non_square_images = 0

min_image_width = 1000000
max_image_width = 0
min_image_height = 1000000
max_image_height = 0

for celebrity in os.listdir(data_dir):
    celebrity_images = os.listdir(os.path.join(data_dir, celebrity))
    num_images = len(celebrity_images)
    total_images += num_images
    avg_images_per_celebrity += num_images
    min_images_per_celebrity = min(min_images_per_celebrity, num_images)
    max_images_per_celebrity = max(max_images_per_celebrity, num_images)
    for image_name in celebrity_images:
        image_path = os.path.join(data_dir, celebrity, image_name)
        image = Image.open(image_path)
        image_width, image_height = image.size
        image_size = os.path.getsize(image_path)
        total_image_size += image_size
        if image_width == image_height:
            num_square_images += 1
        else:
            num_non_square_images += 1
        min_image_width = min(min_image_width, image_width)
        max_image_width = max(max_image_width, image_width)
        min_image_height = min(min_image_height, image_height)
        max_image_height = max(max_image_height, image_height)

avg_images_per_celebrity /= len(celebrities)

print(f"Total celebrities: {len(celebrities)}")
print(f"Total images: {total_images}")
print(f"Average images per celebrity: {avg_images_per_celebrity}")
print(f"Minimum images per celebrity: {min_images_per_celebrity}")
print(f"Maximum images per celebrity: {max_images_per_celebrity}")
print(f"Total image size (bytes): {total_image_size}")
print(f"Number of square images: {num_square_images}")
print(f"Number of non-square images: {num_non_square_images}")
print(f"Minimum image width: {min_image_width}")
print(f"Maximum image width: {max_image_width}")
print(f"Minimum image height: {min_image_height}")
print(f"Maximum image height: {max_image_height}")
