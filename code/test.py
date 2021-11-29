from matplotlib import pyplot as plt
from skimage import io
import os

# image = io.imread('../masks/mask_3.png')
# io.imshow(image) 
# io.show()

path = '../conf'
files= os.listdir(path)
files.sort(key= lambda x:int(x[5:7]))
print(files)
