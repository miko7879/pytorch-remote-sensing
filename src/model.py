import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):

    #Load the pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    
    #Get the number of input features to the original box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    #Define a new box predictor with the requisite number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model