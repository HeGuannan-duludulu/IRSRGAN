
import cv2

x = 70
y = 70

img  = cv2.imread("../1234_srresnet.png")
cv2.rectangle(img, (x, y), (x+128, y+128), (0, 255, 255), 3)
cv2.imshow("1", img)

cv2.waitKey()
cv2.imwrite("./1234_org_full.jpg", img)