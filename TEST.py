import cv2

img_path = r'./test_dir/gt_dir/2.png'
img = cv2.imread(img_path)

gausBlur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Gaussian Blurring', gausBlur)
cv2.waitKey(0)
