import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import json 


# print(np.array([1,2,3]))

img = io.imread('../masks/mask_1.png')
print(img.shape)

# img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
plt.imshow(img)
plt.show()

# # extract the mask
# file = open('../conf/mask_1_indices.json')
# mask_1_indices = json.load(file)

# dest = []
# for key, value in mask_1_indices.items():
#     print(key, value)
#     dest.append(value)
# dest = np.array(dest)
# print(dest.shape)


# image = cv2.imread('../masks/mask_1.png')
# print(image.shape)
# cv2.namedWindow("Image")
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
