import cv2

img = cv2.imread(r'./data/all_img/CVC-09Daytime_000000.png')
print(img.shape)

from imgproc import image_resize

out = image_resize(img, 1/4)

print(out.shape)