import cv2
import numpy as np
import sys
import imageio
from PIL import Image
import PIL.ImageOps

# 0 reads the image in greyscale
with_coin = cv2.imread('images/with_coin.png', 0)
without_coin = cv2.imread('images/without_coin.png', 0)
template = cv2.imread('images/template.png', 0)

np.set_printoptions(threshold=sys.maxsize)
print(with_coin.shape)
print(without_coin.shape)
print(template.shape)

# Distance matrix
distance = np.zeros((210, 210))

for i in range(210):
  for j in range(210):
    
    # Loop over the size of the template and the values from the 'with_coin' matrix
    originalMatrix = np.zeros((40, 40))
    for h in range(40):
      for k in range(40):
        originalMatrix[h][k] = with_coin[i + h][j + k]

    # Calculate Euclidean distance between the two matrices
    dist = np.linalg.norm(originalMatrix - template)
    distance[i][j] = dist

# Normalize matrix to betwen 0 and 255
distance = distance / (distance.max() / 255.0)

imageio.imwrite('images/output_with_coin.png', distance[:, :])
image = Image.open('images/output_with_coin.png')
invert = PIL.ImageOps.invert(image)
invert.save('images/output_with_coin_inverted.png')
