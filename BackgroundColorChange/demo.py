import cv2
import numpy as np
# In this project, we aim to change photo background color to red

# read in a photo
img = cv2.imread('photo.jpg')
# show image on our screen
# cv2.imshow('img', img)
# make the screen hold for some time


# scale the image use resize function

img = cv2.resize(img, None, fx=2, fy=2)
rows, cols, channels = img.shape
print(rows, cols, channels)
# cv2.imshow('img', img)


# change the img to grey img
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', hsv)


# change the photo to binary value
lower_blue = np.array([90, 70, 70])
upper_blue = np.array([110, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)


# decay and dilate
# after make it binary, there are some noise.
erode = cv2.erode(mask, None, iterations=1)
# cv2.imshow('erode', erode)

dilate = cv2.dilate(erode, None, iterations=1)
# cv2.imshow('dilate', dilate)


# iterate every point to substitude the color
for i in range(rows):
    for j in range(cols):
        if erode[i, j] == 255:  # point is white, substitue white
            img[i, j] = (0, 0, 255)  #BRG channel
cv2.imshow('red', img)
cv2.imwrite('result.jpg', img)