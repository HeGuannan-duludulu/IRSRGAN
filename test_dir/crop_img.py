
import cv2
import random

image = cv2.imread('../123.png')
image_height, image_width = image.shape[:2]
image_size = 128

# Just need to find the top and left coordinates of the image
top = random.randint(0, image_height - image_size)
left = random.randint(0, image_width - image_size)

# Crop image patch
patch_image = image[top:top + image_size, left:left + image_size, ...]

cv2.imwrite('./gt_dir/123.png', patch_image)



