import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('resources/2d/image.bmp',0)
img2 = img.copy()
template = cv.imread('resources/2d/template.png',0)
cv.imshow('image', img)
cv.imshow('template', template)
cv.waitKey(0)
w, h = template.shape[::-1]
print(w, h)
#用模板匹配在原图上找到模板位置
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    # 如果方法是 TM_SQDIFF or TM_SQDIFF_NORMED, 取最小值
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()