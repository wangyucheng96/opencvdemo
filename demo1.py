import cv2.cv2 as cv
import numpy as np

POINT = 30


def image_cut(src, max_position):
    res = np.random.randint(10, 99, [500, POINT], np.uint8)
    for i in range(0, 500):
        for j in range(0, POINT):
            res[i, j] = src.item(i+max_position-POINT, j+max_position)
    return res


def max_pos(array):
    return np.argmax(array)


image = cv.imread("data/target.png", 0)
image2 = cv.imread("data/image1.bmp", 0)

print(image.size)
print(image.shape)
print(image.shape[0])   # hang
print(image.shape[1])   # lie
# row1 = image2[0]
# row2 = image2[1]
#
# print(row1)
# print(row2)

# cut = image_cut(image2, 750)
# cv.namedWindow("cut", cv.WINDOW_FREERATIO);
# cv.imshow("cut", cut)
#   visit the value of each pixels by numpy
pic = np.random.randint(10,99,image2.shape, np.uint8)

# for i in range(0,image2.shape[0]):
#     for j in range(0,image2.shape[1]):
#         pic[i,j] = image2.item(i,j)

for i in range(0, image2.shape[0]):
    print(max_pos(image2[i]))

# print(pic)

cv.namedWindow("gray_before",cv.WINDOW_FREERATIO);
cv.imshow("gray_before",image)
cv.namedWindow("original",cv.WINDOW_FREERATIO);
cv.imshow("original",image2)
# for i in range(0,100):
#     for j in range(0,100):
#         print(image[i,j])
#    SOBEL
dst = cv.Sobel(image,-1,1,1)
dst2_x = cv.Sobel(image2,cv.CV_64F,1,0)
dst2_x = cv.convertScaleAbs(dst2_x)
dst2_y = cv.Sobel(image2,cv.CV_64F,0,1)
dst2_y = cv.convertScaleAbs(dst2_y)
#    SOBEL_RESULT
dst_final = cv.addWeighted(dst2_x,0.5,dst2_y,0.5,0)

cv.namedWindow("grad_sobel",cv.WINDOW_FREERATIO);
cv.imshow("grad_sobel",dst_final)
cv.imshow("after",image)
cv.imshow("grad",dst)
#   SCHARR
dst3_x = cv.Scharr(image2,cv.CV_64F,1,0)
dst3_x = cv.convertScaleAbs(dst3_x)
dst3_y = cv.Scharr(image2,cv.CV_64F,0,1)
dst3_y = cv.convertScaleAbs(dst3_y)
#    SCHARR_RESULT
dst_final3 = cv.addWeighted(dst3_x,0.5,dst3_y,0.5,0)

cv.namedWindow("grad_scharr",cv.WINDOW_FREERATIO)
cv.imshow("grad_scharr",dst_final3)

#    LAPLACIAN
lap = cv.Laplacian(image2,cv.CV_64F)
lap = cv.convertScaleAbs(lap)
cv.namedWindow("grad_laplace",cv.WINDOW_FREERATIO)
cv.imshow("grad_laplace",lap)


#    CANNY
canny = cv.Canny(image2,15,128)
cv.namedWindow("grad_canny",cv.WINDOW_FREERATIO)
cv.imshow("grad_canny",canny)
































cv.waitKey()
cv.destroyAllWindows()
