import cv2.cv2 as cv
import numpy as np
import math


def read_img_from_path(img_path):
    original = cv.imread(img_path, 0)
    img = original[0:1080, 3:1919]
    frame = cv.medianBlur(img, 3)
    frame = cv.GaussianBlur(frame, (3, 3), 0)
    frame = cv.bitwise_not(src=img)
    return frame


# retval, threshold_img = cv.threshold(frame, 45, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)


def stretch_gray(frame, index=2):
    e_img = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint16)
    for i in range(e_img.shape[0]):
        for j in range(e_img.shape[1]):
            e_img[i, j] = math.pow(frame[i, j], index)
    cv.normalize(e_img, e_img, 0, 255, cv.NORM_MINMAX)
    gamma_img1 = cv.convertScaleAbs(e_img)
    return gamma_img1


# retval, threshold_img = cv.threshold(gamma_img1, 35, 255, cv.THRESH_BINARY)


def edge_by_dilated(threshold_frame):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # eroded = cv.erode(threshold_img, kernel)
    dilated = cv.dilate(threshold_frame, kernel)

    # res = cv.subtract(threshold_img, eroded)
    absdiff_img = cv.absdiff(dilated, threshold_frame)
    return absdiff_img


# retval, threshold_img = cv.threshold(absdiff_img, 45, 255, cv.THRESH_BINARY)
# retval, threshold_img = cv.threshold(res, 40, 255, cv.THRESH_BINARY)

# if __name__ == '__main__':
#     img_path_path = 'opencv_frame_4_0.png'
#     frame0 = read_img_from_path(img_path_path)
#     frame0 = stretch_gray(frame0, 2)
#     ret, threshold_img = cv.threshold(frame0, 35, 255, cv.THRESH_BINARY)
#     abs_img = edge_by_dilated(threshold_img)
#     # cv.imshow("Eroded Image", eroded)
#     # cv.imshow("Dilated Image", dilated)
#     # cv.imshow("subtract", res)
#     cv.imshow("diff", abs_img)
#     # cv.imshow("t", threshold_img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
