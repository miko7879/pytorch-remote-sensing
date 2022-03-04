import torch

BATCH_SIZE = 4
RESIZE_TO = 1024
EPOCHS = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

CLASSES = ['background', 'Airplane', 'Truncated_airplane']
NUM_CLASSES = 3

TRAIN_DIR = '../data/train/'
VALID_DIR = '../data/valid/'
INF_DIR = '../data/extras/'
OUT_DIR = '../output/'

SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2