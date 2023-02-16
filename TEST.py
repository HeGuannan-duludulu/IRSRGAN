import cv2

org = cv2.imread('123.png')
print(org.shape)

gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
print(gray.shape)