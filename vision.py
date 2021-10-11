import cv2
import numpy as np
import imageio
from PIL import Image
import PIL.ImageOps

# 0 reads the image in greyscale
with_coin = cv2.imread('images/with_coin.png', 0)
without_coin = cv2.imread('images/without_coin.png', 0)
template = cv2.imread('images/template.png', 0)

# Distance matrix
distance_with_coin = np.zeros((210, 210))
distance_without_coin = np.zeros((210, 210))

for i in range(210):
  for j in range(210):
    
    # Loop over the size of the template and the values from the 'with_coin' matrix
    matrix_with_coin = np.zeros((40, 40))
    matrix_without_coin = np.zeros((40, 40))

    for h in range(40):
      for k in range(40):
        matrix_with_coin[h][k] = with_coin[i + h][j + k]
        matrix_without_coin[h][k] = without_coin[i + h][j + k]

    # Calculate Euclidean distance between the two matrices
    dist = np.linalg.norm(matrix_with_coin - template)
    dist2 = np.linalg.norm(matrix_without_coin - template)

    distance_with_coin[i][j] = dist
    distance_without_coin[i][j] = dist2

# Normalize matrix to betwen 0 and 255
distance_with_coin = distance_with_coin / (distance_with_coin.max() / 255.0)
distance_without_coin = distance_without_coin / (distance_without_coin.max() / 255.0)

# Do some IO stuff
imageio.imwrite('images/output_with_coin.png', distance_with_coin[:, :])
imageio.imwrite('images/output_without_coin.png', distance_without_coin[:, :])
image = Image.open('images/output_with_coin.png')
image2 = Image.open('images/output_without_coin.png')
invert = PIL.ImageOps.invert(image)
invert2 = PIL.ImageOps.invert(image2)
invert.save('images/output_with_coin_inverted.png')
invert2.save('images/output_without_coin_inverted.png')
