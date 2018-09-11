from __future__ import division
import random
import math
import numpy as np


def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(128.0, 128.0, 128.0)):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    probability = probability
    mean = mean
    sl = sl
    sh = sh
    r1 = r1

    im_h, im_w, num_ch = img.shape

    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = im_h * im_w

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < im_w and h < im_h:
            x1 = random.randint(0, im_h - h)
            y1 = random.randint(0, im_w - w)
            if num_ch == 3:
                img[x1:x1 + h, y1:y1 + w, :] = np.array(mean).reshape([1, 1, 3])
            else:
                img[x1:x1 + h, y1:y1 + w] = mean[0]
            return img

    return img
