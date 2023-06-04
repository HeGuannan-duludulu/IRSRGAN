"""
Random crop single image, the crop size is 128x128
"""


import cv2
import random

input_img = '../your_input_img.png'
output_img = './your_output_img.png'

image = cv2.imread(input_img)
image_height, image_width = image.shape[:2]
image_size = 128

top = random.randint(0, image_height - image_size)
left = random.randint(0, image_width - image_size)

# Crop image patch
patch_image = image[top:top + image_size, left:left + image_size, ...]

# save the output img
cv2.imwrite(output_img, patch_image)



