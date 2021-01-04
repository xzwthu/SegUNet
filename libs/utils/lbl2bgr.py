import numpy as np


color_dict = {
      0: (  0,   0,   0),
      1: (128,   0,   0),  2: (  0, 128,   0),  3: (128, 128,   0),  4: (  0,   0, 128),
      5: (128,   0, 128),  6: (  0, 128, 128),  7: (128, 128, 128),  8: ( 64,   0,   0),
      9: (192,   0,   0), 10: ( 64, 128,   0), 11: (192, 128,   0), 12: ( 64,   0, 128),
     13: (192,   0, 128), 14: ( 64, 128, 128), 15: (192, 128, 128), 16: (  0,  64,   0),
     17: (128,  64,   0), 18: (  0, 192,   0), 19: (128, 192,   0), 20: (  0,  64, 128),
    255: (255, 255, 255)
}


def lbl2bgr(mask):
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color = color_dict[mask[i, j]]
            for k in range(3):
                img[i, j, k] = color[3 - k - 1]

    return img