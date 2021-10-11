import cv2
import numpy as np
import sys

# 0 reads the image in greyscale
with_coin = cv2.imread('images/with_coin.png', 0)
without_coin = cv2.imread('images/without_coin.png', 0)
template = cv2.imread('images/template.png', 0)

np.set_printoptions(threshold=sys.maxsize)
print(with_coin.shape)
print(without_coin.shape)
print(template.shape)
#print(template)

