import cv2 as cv
import numpy as np
from PIL import Image
from random import randint
import sys


KERNEL_NOISE_SIZE = 3
noise_mask = np.ones((KERNEL_NOISE_SIZE, KERNEL_NOISE_SIZE), dtype=np.float32)
noise_mask[KERNEL_NOISE_SIZE//2][KERNEL_NOISE_SIZE//2] = 0


def processing(filename):
    image = cv.imread(filename)
    gr = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gr = cv.filter2D(gr, -1, noise_mask)
    kernel_open_size = np.max(gr.shape) // 40
    kernel_erosion_size = np.max(gr.shape) // 65
    kernel_gradient_size = np.max(gr.shape) // 100
    iterations = 2
    open_kernel = np.ones((kernel_open_size, kernel_open_size), np.uint8)
    erosion_kernel = np.ones((kernel_erosion_size, kernel_erosion_size), np.uint8)
    gradient_kernel = np.ones((kernel_gradient_size, kernel_gradient_size), np.uint8)
    opening = cv.morphologyEx(gr, cv.MORPH_OPEN, open_kernel)
    erosion = cv.erode(opening, erosion_kernel, iterations=iterations)
    gradient = cv.morphologyEx(erosion, cv.MORPH_GRADIENT, gradient_kernel)
    closing = cv.morphologyEx(gradient, cv.MORPH_CLOSE, open_kernel)
    save_img = Image.fromarray(closing)
    rand_name = f"./processed_images/{randint(0, 100)}.png"
    save_img.save(rand_name)
    print(f"Image saved as {rand_name}")

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        filename = args[1]
        try:
            processing(filename)
        except BaseException:
            print("Error...")

    else:
        pass
