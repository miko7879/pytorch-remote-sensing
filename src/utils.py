import albumentations as A
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes

class Averager:

    def __init__(self):
        self.total = 0
        self.count = 0
        
    def send(self, val):
        self.total += val
        self.count += 1
        
    def value(self):
        if self.count == 0:
            return 0
        return self.total/self.count
        
    def reset(self):
        self.total = 0
        self.count = 0
        
def collate_fn(batch):
    return tuple(zip(*batch)) #The'*' operator allows us to accept an arbitrary number of arguments
    
def get_train_transform():
    return A.Compose([
        A.Flip(p = 0.5), #Flip with a probability of 50%
        A.RandomRotate90(p = 0.5),
        A.MotionBlur(p = 0.2),
        A.MedianBlur(blur_limit = 3, p = 0.1), #blur_limit is the maximum kernel/aperature size
        A.Blur(blur_limit = 3, p = 0.1),
        ToTensorV2(p = 1.0) #Transform to Tensor
    ], bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
    
def get_valid_transform():
    return A.Compose([ #Just transform to tensor, make appropriate bounding box changes, and return
    ToTensorV2(p = 1.0)
    ], bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })