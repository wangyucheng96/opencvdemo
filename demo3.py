# create an image to emulate the cross line image
# author : Wyc
# date : 12.13

import cv2.cv2 as cv
import numpy as np


def create_img(height, width, mid_width, max_pixel_value):
    res = np.zeros([height, width, 1], np.uint8)
    mid1 = width//2
    mid2 = height//2

    start1 = mid1 - mid_width//2
    end1 = mid1 + mid_width//2

    start2 = mid2 - mid_width//2
    end2 = mid2 + mid_width//2

    for row in range(start2, end2):
        for col in range(width):
            res[row, col] = max_pixel_value

    for row in range(height):
        for col in range(start1, end1):
            res[row, col] = max_pixel_value

    cv.namedWindow("res")
    cv.resizeWindow("res", 800, 600)
    cv.imshow("res", res)
    return res


res0 = create_img(6000, 8000, 100, 210)

cv.imwrite("data/res_with_max_pixel_val_210.png", res0)


cv.waitKey()
cv.destroyAllWindows()