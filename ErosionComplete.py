# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:24:09 2019

@author: Mohammadreza
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt



image = plt.imread('TestErosion.jpg', format=None)
image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
res, image = cv2.threshold(image,127,1,1)
#plt.imshow(image, cmap='gray')
#plt.figure()

image = np.array([[1,0,1,0,1],[0,1,1,1,0],[0,0,1,0,1]])
#print(image)
dimensions = image.shape
height = image.shape[0]
width = image.shape[1]
#print(image)

#S = (1,3) 
#B = np.ones(S,dtype =int)
B=np.array([[0,1,0],[1,1,1],[0,1,0]])
#B=np.array([[1,1,1],[]])
#print(B)


D0 = np.zeros((height,width), dtype=int)
#print(D0)

a11=np.array(np.pad(image, ((1,1),(1,1)), 'constant'))
#print(a11)
#print(a11)

H = a11.shape[0]
W = a11.shape[1]

#B = np.flip(B,(1))
#B = np.flip(B,(0))


for row in range(height):
    for col in range(width):
        #print('o')
        Test =(np.array_equal(a11[row:(row+B.shape[0]),col:(col+B.shape[1])],B))
        #print(Test)
        if Test == True:
            D0[row][col] = 1
           


plt.imshow(image, cmap='gray')
plt.figure()
plt.imshow(D0, cmap='gray')
plt.show()
