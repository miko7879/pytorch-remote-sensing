import torch

#Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

#ML Parameters
BATCH_SIZE = 4
RESIZE_TO = 1024
EPOCHS = 100
LR = 0.001
MOMENTUM = 0.99
DETECTION_THRESHOLD = 0.7
NUM_CLASSES = 3
CLASSES = ['background', 'Airplane', 'Truncated_airplane']

#Directories
TRAIN_DIR = '../data/train/'
VALID_DIR = '../data/valid/'
INF_DIR = '../data/extras/'
OUT_DIR = '../output/'

#Save frequencies
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2