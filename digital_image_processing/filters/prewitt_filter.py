import numpy as np
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey

from convolve import img_convolve

def prewitt_filter(image):
    kernel_x = np.array([[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]])
    kernel_y = np.array([[1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]])
    dst_x = np.abs(img_convolve(image, kernel_x))
    dst_y = np.abs(img_convolve(image, kernel_y))
    dst_x = dst_x * 255 / np.max(dst_x)
    dst_y = dst_y * 255 / np.max(dst_y)

    dst_xy = np.sqrt((np.square(dst_x)) + (np.square(dst_y)))
    dst_xy = dst_xy * 255 / np.max(dst_xy)
    dst = dst_xy.astype(np.uint8)

    theta = np.arctan2(dst_y, dst_x)
    return dst, theta


if __name__ == "__main__":
    img = imread("../image_data/lena.jpg")
    gray = cvtColor(img, COLOR_BGR2GRAY)
    prewitt_grad, prewitt_theta = prewitt_filter(gray)
    imshow("Prewitt grad", prewitt_grad)
    imshow("Prewitt theta", prewitt_theta)
    waitKey(0)