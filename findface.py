#!/usr/bin/python3

import cv2
import numpy as np

def arrowbox(img, x,y,w,h, clr):
    cv2.rectangle(img, (x, y), (x+w, y+h), clr, 8)
    top = (x,y)
    bottom = (x,y+h)
    middle = (int(x - w/4), int(y + h/2))
    triangle = np.array([top, bottom, middle])
    cv2.drawContours(img, [triangle], 0, clr, -1)


img = cv2.imread("crowd.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 4)

face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    arrowbox(img, x,y,w,h,(0,255,0))

gray = cv2.flip(gray,1)
img = cv2.flip(img,1)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    arrowbox(img, x,y,w,h,(0,255,0))

img = cv2.flip(img,1)

h,w = img.shape[:2]

h = int(800* h/w)
w = 800

showimg = cv2.resize(img,(w,h))

cv2.imshow("image", showimg)
cv2.waitKey(5000)






