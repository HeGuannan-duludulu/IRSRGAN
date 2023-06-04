"""
Draw yellow square on image
"""

import cv2


def draw_square(org_img="../1234_srresnet.png",
                output_img="./have_yellow_squared.png",
                square_size=128,
                x=70,
                y=70):
    """
    Draw yellow square on image
    :param org_img:
    :param output_img:
    :param square_size:
    :param x:
    :param y:
    :return:
    """
    img = cv2.imread(org_img)
    cv2.rectangle(img, (x, y), (x + square_size, y + square_size), (0, 255, 255), thickness=3)
    cv2.imshow("output_img", img)
    cv2.waitKey()
    cv2.imwrite(output_img, img)
