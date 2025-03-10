from __future__ import absolute_import

from torchvision.transforms import *

import random
import math
import torch
import numpy as np

from .operations import *


class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    
    Args:
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                return img

        return img


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, probability):
        self.n_holes = n_holes
        self.length = length
        self.probability = probability

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if random.uniform(0, 1) > self.probability:
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# class AutoAugment(object):
    
#     def __init__(self):
#         pass

#     def __call__(self, img):
#         """
#         Args:
#             img (Tensor): Tensor image of size (C, H, W).
#         """
#         policies = {'Policy_0':  [Posterize(0.4, 8),    Rotate(0.6,9)],
#                     'Policy_1':  [Solarize(0.6, 5),     AutoContrast(0.6, 5)],
#                     'Policy_2':  [Equalize(0.8, 8),     Equalize(0.6, 3)], 
#                     'Policy_3':  [Posterize(0.6, 7),    Posterize(0.6, 3)],
#                     'Policy_4':  [Equalize(0.4, 7),     Solarize(0.2, 4)],
#                     'Policy_5':  [Equalize(0.4, 4),     Rotate(0.8, 8)],
#                     'Policy_6':  [Solarize(0.6, 3),     Equalize(0.6, 7)],
#                     'Policy_7':  [Posterize(0.8, 5),    Equalize(1.0, 2)],
#                     'Policy_8':  [Rotate(0.2, 3),       Solarize(0.6, 8)],
#                     'Policy_9':  [Equalize(0.6, 8),     Posterize(0.4, 6)],
#                     'Policy_10': [Rotate(0.8, 8),       Color(0.4, 0)],
#                     'Policy_11': [Rotate(0.4, 9),       Equalize(0.6, 2)],
#                     'Policy_12': [Equalize(0.0, 7),     Equalize(0.8, 8)],
#                     'Policy_13': [Invert(0.6, 4),       Equalize(1.0, 8)],
#                     'Policy_14': [Color(0.6, 4),        Contrast(1.0, 8)],
#                     'Policy_15': [Rotate(0.8, 8),       Color(1.0, 2)],
#                     'Policy_16': [Color(0.8, 8),        Solarize(0.8, 7)],
#                     'Policy_17': [Sharpness(0.4, 7),    Invert(0.6, 8)],
#                     'Policy_18': [ShearX(0.6, 5),       Equalize(1.0, 9)],
#                     'Policy_19': [Color(0.4, 0),        Equalize(0.6, 3)],
#                     'Policy_20': [Equalize(0.4, 7),     Solarize(0.2, 4)],
#                     'Policy_21': [Solarize(0.6, 5),     AutoContrast(0.6, 5)],
#                     'Policy_22': [Invert(0.6, 4),       Equalize(1.0, 8)],
#                     'Policy_23': [Color(0.6, 4),        Contrast(1.0, 8)],
#                     'Policy_24': [Equalize(0.8, 8),     Equalize(0.6, 3)],
#                     }

                    

#         policy = random.choice([policies['Policy_%d'%i] for i in range(25)])

#         for op in policy:
#             img = op(img)

#         return img


from .augmentation_transforms import *

class AutoAugment(object):
    
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        """

        #
        # ImageNet policies proposed in https://arxiv.org/abs/1805.09501
        #
        #policies = [
        #    [('Posterize', 0.4, 8),    ('Rotate', 0.6,9)],
        #    [('Solarize', 0.6, 5),     ('AutoContrast', 0.6, 5)],
        #    [('Equalize', 0.8, 8),     ('Equalize', 0.6, 3)], 
        #    [('Posterize', 0.6, 7),    ('Posterize', 0.6, 3)],
        #    [('Equalize', 0.4, 7),     ('Solarize', 0.2, 4)],
        #    [('Equalize', 0.4, 4),     ('Rotate', 0.8, 8)],
        #    [('Solarize', 0.6, 3),     ('Equalize', 0.6, 7)],
        #    [('Posterize', 0.8, 5),    ('Equalize', 1.0, 2)],
        #    [('Rotate', 0.2, 3),       ('Solarize', 0.6, 8)],
        #    [('Equalize', 0.6, 8),     ('Posterize', 0.4, 6)],
        #    [('Rotate', 0.8, 8),       ('Color', 0.4, 0)],
        #    [('Rotate', 0.4, 9),       ('Equalize', 0.6, 2)],
        #    [('Equalize', 0.0, 7),     ('Equalize', 0.8, 8)],
        #    [('Invert', 0.6, 4),       ('Equalize', 1.0, 8)],
        #    [('Color', 0.6, 4),        ('Contrast', 1.0, 8)],
        #    [('Rotate', 0.8, 8),       ('Color', 1.0, 2)],
        #    [('Color', 0.8, 8),        ('Solarize', 0.8, 7)],
        #    [('Sharpness', 0.4, 7),    ('Invert', 0.6, 8)],
        #    [('ShearX', 0.6, 5),       ('Equalize', 1.0, 9)],
        #    [('Color', 0.4, 0),        ('Equalize', 0.6, 3)],
        #    [('Equalize', 0.4, 7),     ('Solarize', 0.2, 4)],
        #    [('Solarize', 0.6, 5),     ('AutoContrast', 0.6, 5)],
        #    [('Invert', 0.6, 4),       ('Equalize', 1.0, 8)],
        #    [('Color', 0.6, 4),        ('Contrast', 1.0, 8)],
        #    [('Equalize', 0.8, 8),     ('Equalize', 0.6, 3)],
        #]

        # Cifar policies
        policies = [
            [('Invert', 0.1, 7),        ('Contrast', 0.2, 6)],
            [('Rotate', 0.7, 2),        ('TranslateX', 0.3, 9)],
            [('Sharpness', 0.8, 1),     ('Sharpness', 0.9, 3)],
            [('ShearY', 0.5, 8),        ('TranslateY', 0.7, 9)],
            [('AutoContrast', 0.5, 8),  ('Equalize', 0.9, 2)],

            [('ShearY', 0.2, 7),        ('Posterize', 0.3, 7)],
            [('Color', 0.4, 3),         ('Brightness', 0.6, 7)],
            [('Sharpness', 0.3, 9),     ('Brightness', 0.7, 9)],
            [('Equalize', 0.6, 5),      ('Equalize', 0.5, 1)],
            [('Contrast', 0.6, 7),      ('Sharpness', 0.6, 5)],

            [('Color', 0.7, 7),         ('TranslateX', 0.5, 8)],
            [('Equalize', 0.3, 7),      ('AutoContrast', 0.4, 8)],
            [('TranslateY', 0.4, 3),    ('Sharpness', 0.2, 6)],
            [('Brightness', 0.9, 6),    ('Color', 0.2, 8)],
            [('Solarize', 0.5, 2),      ('Invert', 0.0, 3)],

            [('Equalize', 0.2, 0),      ('AutoContrast', 0.6, 0)],
            [('Equalize', 0.2, 8),      ('Equalize', 0.8, 4)],
            [('Color', 0.9, 9),         ('Equalize', 0.6, 6)],
            [('AutoContrast', 0.8, 4),  ('Solarize', 0.2, 8)],
            [('Brightness', 0.1, 3),    ('Color', 0.7, 0)],

            [('Solarize', 0.4, 5),      ('AutoContrast', 0.9, 3)],
            [('TranslateY', 0.9, 9),    ('TranslateY', 0.8, 9)],
            [('AutoContrast', 0.9, 2),  ('Solarize',0.8, 3)],
            [('Equalize', 0.8, 8),      ('Invert', 0.1, 3)],
            [('TranslateY', 0.7, 9),    ('AutoContrast', 0.9, 1)],
        ]


        policy = random.choice(policies)

        img = apply_policy(policy, img)

        return img.convert('RGB')


class FastAutoAugment(object):

        #
        # ImageNet policies proposed in https://arxiv.org/abs/1905.00397
        #
    

    def __init__(self):

        from .archive import fa_reduced_cifar10

        self.policies = fa_reduced_cifar10()
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        """

        policy = random.choice(self.policies)

        img = apply_policy(policy, img)

        return img.convert('RGB')
