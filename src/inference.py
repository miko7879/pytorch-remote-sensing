import numpy as np
import cv2
import torch
import glob as glob

from model import create_model
from config import CLASSES, NUM_CLASSES, OUT_DIR, INF_DIR, DEVICE, DETECTION_THRESHOLD

#Create the model and load the trained parameters
model = create_model(num_classes = NUM_CLASSES)
model.load_state_dict(torch.load(OUT_DIR + 'model_final.pth'))
model.eval()
model.to(DEVICE)

#Grab all files from the inference directory
infer_imgs = glob.glob(f'{INF_DIR}*')
print(f'Number if images to inference on: {len(infer_imgs)}')

#Lastly, let's loop over our images
for i in range(len(infer_imgs)):
    
    img_name = infer_imgs[i].replace("\\","/").split('/')[-1].split('.')[0] #Get image name
    img = cv2.imread(infer_imgs[i]) #Read in the image
    orig_img = img.copy() #Make a copy to draw bounding boxes on
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) #Convert image to appropriate format and pixel values to numpy float32
    img = img/255.0 #Normalize image
    img = np.transpose(img, (2, 0, 1)).astype(np.float)#Place colour channel first, then height, then width
    img = torch.tensor(img, dtype = torch.float) #Convert image to tensor
    img = torch.unsqueeze(img, 0) #Add batch dimension
    img = img.to(DEVICE) #Push to device
    
    #Get predictions
    with torch.no_grad():
        preds = model(img)
    
    #Load all detections to CPU for processing
    preds = {k : v.to(torch.device('cpu')) for k, v in preds[0].items()}
    
    #If we found something, let's draw it on our image
    if len(preds['boxes']) != 0:
        
        #Extract the boxes
        boxes = preds['boxes'].data.numpy()
        scores = preds['scores'].data.numpy()
        boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32) #Extract valid bounding boxes and convert to requisite type
        draw_boxes = boxes.copy() #Make a copy for further manipulation
        
        #Extract the labels
        labels = [CLASSES[j] for j in preds['labels'].cpu().numpy()]
        
        #Draw the bounding boxes and label them
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(orig_img, labels[j], (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        
    #Write the image
    cv2.imwrite(f'{OUT_DIR}inference/{img_name}.jpg', orig_img,)
        
    print(f'Image {i + 1} complete')