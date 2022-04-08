import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('data/new_5m5.png', 0)
height = img.shape[0]//5
width = img.shape[1]//5
print(height)
print(width)

res = np.zeros([height, width, 1], np.uint8)

for i in range(0, 5):
    for row in range(height):
        for col in range(width):
            res[row, col] = img[row*5+i, col*5+i]
    print("NO."+str(i))
    cv.imshow("res", res)
    cv.imwrite("data/final_new_5m5_"+str(i)+".png", res)


cv.waitKey()
cv.destroyAllWindows()