import numpy as np
import json
import cv2


def preprocess(face, mask, mask_indices):
    indices = np.array([10, 152, 234, 454, 159, 145, 33, 133, 386, 374, 263, 362, 1, 13])
    face = np.array(face)
    dest = face[indices]
    face = list(face)
    dest = np.array(dest, dtype="float32")
    
    # put the label indices of mask in the src[]
    mask_image = cv2.imread(mask, cv2. IMREAD_UNCHANGED)
    file = open(mask_indices)
    mask_1_indices = json.load(file)
    
    src = []
    for key, value in mask_1_indices.items():
        src.append(value)
    src = np.array(src, dtype="float32")
    
    cv2.destroyAllWindows()

    return dest, src, mask_image
