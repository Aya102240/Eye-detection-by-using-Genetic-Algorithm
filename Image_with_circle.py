# _______________________________________________________________________________

# import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# _______________________________________________________________________________

# get the original image
image_1 = Image.open("Original/img2.jpg")
# get width and height of the original image
width, height = image_1.size

# _______________________________________________________________________________

# create new image with a black color only
img = np.zeros((height, width, 3), dtype = np.uint8)
# draw circle with white edge at the same location of 1'st eye with same radius
cv2.circle(img, (114, 183), 8, (255, 255, 255))
cv2.imwrite("test.jpg", img)

# _______________________________________________________________________________

# original image (upload image)
image = cv2.imread("Original/img2.jpg")
# make image in a grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Blurring the image by Gaussian Blur to remove background image
imageWithGauBlur = cv2.GaussianBlur(gray, (5, 5), 0)

# _______________________________________________________________________________

# Edge detection by canny method
def Canny_edge_original_img():
    edges = cv2.Canny(imageWithGauBlur, 100, 200)
    return np.float16(edges)

# _______________________________________________________________________________

def desired_output(rad):
    out = unknowncircle(rad) * Canny_edge_original_img()
    non_zero = np.nonzero(out)
    length_of_n_of_elements = len(non_zero[0])
    return length_of_n_of_elements

# _______________________________________________________________________________

def unknowncircle(radius):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (114, 183), radius, (255, 255, 255))
    cv2.Canny(img, 100, 200)
    cv2.imwrite("test.jpg", img)
    nonzero = np.nonzero(img)
    length_of_num_of_elements = len(nonzero[1])
    return length_of_num_of_elements

# _______________________________________________________________________________

def draw_circle(rad):
    img = cv2.imread("Original/img2.jpg")
    cv2.circle(img, (114, 183), int(rad), (200, 220, 0), 1)
    cv2.imwrite("First_eye.jpg", img)

# _______________________________________________________________________________

def second_circle(rad):
    img = cv2.imread("First_eye.jpg")
    cv2.circle(img, (204, 182), int(rad), (200, 220, 0), 1)
    cv2.imshow("======Result======", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# _______________________________________________________________________________