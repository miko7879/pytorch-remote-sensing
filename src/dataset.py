import torch
import cv2
import numpy as np
import os
import glob
import csv

from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform

class AirplaneDataset(Dataset):

    def __init__(self, dir_path, height, width, classes, transforms = None):
    
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        
        self.image_paths = glob.glob(f'{self.dir_path}*.jpg') #Grab all images in assigned path
        self.all_images = [p.replace('\\', '/').split('/')[-1] for p in self.image_paths]
        self.all_images = sorted(self.all_images)
        
        self.class_dict = {}
        for i in range(1, len(classes)):
            self.class_dict[classes[i]] = i
            
        #Grab all annotations, store them in memory for quick access
        tmp = set(self.all_images)
        
        with open('../data/annotations.csv', newline='') as f:
            reader = csv.reader(f)
            l = list(reader)
            self.annotations = {img : [annot[2:] for annot in l if annot[1] == img] for img in tmp}
            
    def __len__(self):
        return len(self.all_images)
        
    def __getitem__(self, idx):
        
        #Reconstruct image path
        iname = self.all_images[idx]
        ipath = os.path.join(self.dir_path, iname)
        
        #Read in the image
        img = cv2.imread(ipath)
        
        #Convert to appropriate channel order and set the datatype to float (it was previously uint)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        #Resize and normalize image
        img_resized = cv2.resize(img, (self.width, self.height))
        img_resized = img_resized/255.0
        
        #Create empty lists to store the bounding boxes and labels
        boxes = []
        labels = []
        
        #Grab the original height and width of the image, will be used to resize bouding boxes
        iwidth = img.shape[1]
        iheight = img.shape[0]
        
        #Iterate over each bounding box
        for coords, cls in self.annotations[iname]:
            
            #Extract the coordinates
            coords = coords[1:-1].replace(' ', '').split(',')
            xmin, ymin, xmax, ymax = int(coords[0][1:]), int(coords[1][:-1]), min(int(coords[4][1:]), iwidth), min(int(coords[5][:-1]), iheight)
            
            #Adjust bounding box coordinates
            xmin = xmin/iwidth*self.width
            ymin = ymin/iheight*self.height
            xmax = xmax/iwidth*self.width
            ymax = ymax/iheight*self.height
            
            if xmax > 1024 or ymax > 1024:
                print(iname, xmin, ymin, xmax, ymax, cls)
            
            #Add to our list
            boxes.append([xmin, ymin, xmax, ymax])
            
            #Append label to list
            labels.append(self.class_dict[cls])
            
        #Convert to tensors of appropriate data type
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype = torch.int64)
        area = ((boxes[:, 3] - boxes[:, 1])*(boxes[:, 2] - boxes[:, 0]))
        iscrowd = torch.zeros((boxes.shape[0], ), dtype = torch.int64)
        img_id = torch.tensor([idx])
        
        #Prepare the final dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrows'] = iscrowd
        target['image_id'] = img_id
        
        #Lastly, let's apply the transforms and return the transformed image along with the transformed target dictionary
        if self.transforms:
            transf = self.transforms(image = img_resized, bboxes = target['boxes'], labels = labels)
            img_resized = transf['image'] #Grab transformed image
            target['boxes'] = torch.Tensor(transf['bboxes']) #Grab transformed bounding boxes
            
        return img_resized, target


#Instantiate Dataset instances
train_dataset = AirplaneDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = AirplaneDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

#Instantiate DataLoader instances
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, collate_fn = collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0, collate_fn = collate_fn)

#Give some console output on the dataset lengths
print('')
print(f'Number of training images: {len(train_dataset)}')
print(f'Number of validation images: {len(valid_dataset)}')
print('') 
        
if __name__ == '__main__':
    
    #Instantiate a dataset
    dataset = AirplaneDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES)
    
    #Function to visualize a single sample
    def visualize_sample(image, target):
    
        boxes = target['boxes']
        labels = target['labels']
        
        for i in range(len(boxes)):
            box, label = boxes[i], CLASSES[labels[i]]
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
            cv2.putText(image, label, (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)