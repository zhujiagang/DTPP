import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np
lena = mpimg.imread('TIM.png')
print lena.shape

img = lena[:,:,0:3]
plt.imshow(img)
plt.axis('off')
plt.show()
print img.shape

lena = mpimg.imread('yulan.jpg')
print lena.shape
plt.imshow(lena)
plt.axis('off')
#plt.show()
lena = lena.transpose(2,0,1)
print lena.shape
lena[0,:,:] = lena[0,:,:] - 104
lena[1,:,:] = lena[1,:,:] - 117
lena[2,:,:] = lena[2,:,:] - 123
#lena = lena.transpose(1,2,0)
#plt.imshow(lena)
plt.axis('off')
plt.show()