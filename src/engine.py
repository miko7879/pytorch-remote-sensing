from config import DEVICE, NUM_CLASSES, EPOCHS, OUT_DIR, SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from utils import Averager
from dataset import train_loader, valid_loader
from model import create_model
from tqdm.auto import tqdm

import torch
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')

def train(train_data_loader, model, optim, train_loss_list, train_loss_hist):

    print('Training')
    
    #Keep track of total iterations
    global train_iter
    
    #Instatiate tqdm progress bar
    prog_bar = tqdm(train_data_loader, total = len(train_data_loader))
    
    #Iterate over each batch
    for images, targets in prog_bar:
    
        #Clear the gradient before forward pass
        optim.zero_grad()
        
        #Push data to device 
        images = [image.to(DEVICE) for image in images]
        targets = [{k : v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        #Forward pass
        loss_dict = model(images, targets)
        
        #Calculate the total loss in this batch
        losses = sum(loss for loss in loss_dict.values())
        loss_val = losses.item()
        
        #Append batch loss to the list of losses for plotting and send to averager to calculate epoch loss
        train_loss_list.append(loss_val)
        train_loss_hist.send(loss_val)
        
        #Perform backward step and update weights
        losses.backward()
        optim.step()
        
        #Update training iterations
        train_iter += 1
        
        #Update the progrss bar
        prog_bar.set_description(desc = f'Loss: {loss_val:.4f}')
        
def valid(valid_data_loader, model, valid_loss_list, valid_loss_hist):

    print('Validating')
    
    #Keep track of total iterations
    global valid_iter
    
    #Instatiate tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total = len(valid_data_loader))
    
    for images, targets in prog_bar:
    
        #Push data to device
        images = [image.to(DEVICE) for image in images]
        targets = [{k : v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        #Foward pass, no gradient
        with torch.no_grad():
            loss_dict = model(images, targets)
            
        #Calculate total loss in batch
        losses = sum(loss for loss in loss_dict.values())
        loss_val = losses.item()
        
        #Append batch loss to list of losses and send to averager
        valid_loss_list.append(loss_val)
        valid_loss_hist.send(loss_val)
        
        #Update valid iterations
        valid_iter += 1
        
        #Update the progress bar
        prog_bar.set_description(desc = f'Loss: {loss_val:.4f}')
        
        
if __name__ == '__main__':

    #Instantiate model and push to device
    model = create_model(num_classes = NUM_CLASSES)
    model.to(DEVICE)
    
    print(f'Using device: {DEVICE}')
    
    #Extract requisite parameters and instantiate optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr = 0.001, momentum = 0.99)
    
    #Initialize helper objects, lists, and counters
    train_loss_hist = Averager()
    valid_loss_hist = Averager()
    train_iter = 1
    valid_iter = 1
    train_loss_list = []
    valid_loss_list = []
    
    #Define the model name
    MODEL_NAME = 'model'
    
    #Train for a given number of epochs:
    for e in range(EPOCHS):
    
        print(f'\nEPOCH {e + 1} / {EPOCHS}\n')

        #Reset the training and validation averagers
        train_loss_hist.reset()
        valid_loss_hist.reset()
        
        #Create two subplots, one for training and one for validation
        f1, train_ax = plt.subplots()
        f2, valid_ax = plt.subplots()
        
        #Start the timer
        stime = time.time()
        
        #Training
        train(train_loader, model, optim, train_loss_list, train_loss_hist)
        
        #Validation
        valid(valid_loader, model, valid_loss_list, valid_loss_hist)
        
        #Display the training and validation loss for this epochs
        print(f'\nEpoch training loss: {train_loss_hist.value():.3f}')
        print(f'Epoch validation loss: {valid_loss_hist.value():.3f}')
        
        #Complete timing
        print(f'Epoch time: {time.time() - stime}')
        
        #If required, save model
        if (e + 1) % SAVE_MODEL_EPOCH == 0:
            torch.save(model.state_dict(), f'{OUT_DIR}/model_e{e + 1}.pth')
            print('SAVING MODEL COMPLETE')
            
        #If required, save plot
        if (e + 1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss_list, color = 'blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            f1.savefig(f'{OUT_DIR}/train_loss_e{e + 1}.png')
            valid_ax.plot(valid_loss_list, color = 'red')
            valid_ax.set_xlabel('iterations')
            train_ax.set_ylabel('validation loss')
            f2.savefig(f'{OUT_DIR}/valid_loss_e{e + 1}.png')
            print('SAVING PLOTS COMPLETE')
            
        #Save at least once at the end
        if (e + 1) == EPOCHS:
            
            #Save model
            torch.save(model.state_dict(), f'{OUT_DIR}/model_final.pth')
            
            #Save plots
            train_ax.plot(train_loss_list, color = 'blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            f1.savefig(f'{OUT_DIR}/train_loss_final.png')
            valid_ax.plot(valid_loss_list, color = 'red')
            valid_ax.set_xlabel('iterations')
            train_ax.set_ylabel('validation loss')
            f2.savefig(f'{OUT_DIR}/valid_loss_final.png')
            
        #Close all plots
        plt.close('all')