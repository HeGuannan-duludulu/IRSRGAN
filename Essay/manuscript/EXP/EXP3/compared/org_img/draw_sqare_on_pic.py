


import cv2

# 读取图像
name = '0643.jpg'
img = cv2.imread(name)

# 定义矩形的左上角和右下角坐标
x1, y1 = 48, 350
x2, y2 = x1+128, y1+128

# 绘制矩形
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

# 显示图像
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./drew/{}".format(name), img)
