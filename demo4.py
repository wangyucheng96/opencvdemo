import cv2.cv2 as cv
import numpy as np

image = cv.imread('data/res_with_max_pixel_val_210.png', 0)

# cv.namedWindow("core", cv.WINDOW_FREERATIO)
# cv.imshow("core", core)
#
# cv.namedWindow("image", cv.WINDOW_FREERATIO)
# cv.imshow("image", image)

# new = np.zeros([400, 400, 1], np.uint8)


# 高斯滤波
gaussian = cv.GaussianBlur(image, (99, 99), 25)
cv.namedWindow("guss", 0)
cv.resizeWindow("guss", 800, 600)
cv.imshow("guss", gaussian)

# 均值滤波
# a window with the size of 5*5
new1 = cv.blur(gaussian, (5, 5))
cv.namedWindow("mean", 0)
cv.resizeWindow("mean", 800, 600)
cv.imshow("mean", new1)


def guss_noise(src, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    # src = np.array(src/255)
    noise = np.random.normal(mean, var ** 0.5, src.shape)
    out = src + noise
    # if out.min() < 0:
    #     low_clip = -1.
    # else:
    #     low_clip = 0.
    # out = np.clip(out, low_clip, 1.0)
    # out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out


final1 = guss_noise(new1, 0, 9)
cv.namedWindow("final", 0)
cv.resizeWindow("final", 800, 600)
cv.imshow("final", final1)

# final2 = guss_noise(new1, 0, 0.02)

# final3 = cv.blur(final1, (10, 10))

# cv.imshow("guss_res_with_guss_noise", final1)
# cv.imshow("mean_res_with_guss_noise", final2)
# cv.imshow("guss_res_with_guss_noise_with_low_pass", final3)
cv.imwrite("data/new_5m5.png", final1)
# cv.imwrite("data/mean_noise.png", final2)
# cv.imwrite("data/guss_mean_noise.png", final3)

cv.waitKey()
cv.destroyAllWindows()