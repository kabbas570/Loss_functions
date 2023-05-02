import numpy as np

class EarlyStopping:
    def __init__(self, patience=None, verbose=True, delta=0,  trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score1 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.max_score = 0
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss,val_metric):

        score = -val_loss
        score1=-val_metric

        if (self.best_score is None) and (self.best_score1 is None):
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
        elif (score < self.best_score + self.delta) or (score1 > self.best_score1 + self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
            self.counter = 0

    def verbose_(self, val_loss,val_metric):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.trace_func(f'Validation metric increased ({self.max_score:.6f} --> {val_metric:.6f}).')
        self.val_loss_min = val_loss
        self.max_score = val_metric
        
        ### this should be in the main taining 
        
 from Early_Stopping import EarlyStopping
early_stopping = EarlyStopping(patience=4, verbose=True)


     
epoch_len = len(str(Max_Epochs))

path_to_save_Learning_Curve='/data/scratch/acw676/VAE_weights/'+'/NO_LINEAR_2D_ES_Simoid1'
path_to_save_check_points='//data/scratch/acw676/VAE_weights/'+'/NO_LINEAR_2D_ES_Simoid1'  ##these are weights

### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
 
from Model2_2D2 import Autoencoder_2D2_8
model_1 = Autoencoder_2D2_8()
  
def main():
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
    optimizer1 = optim.AdamW(model1.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    loss_function = combine_loss
    for epoch in range(Max_Epochs):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model1, optimizer1,scaler,loss_function)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        dice_score = check_Acc(val_loader, model1,loss_function, device=DEVICE)
        avg_valid_DS1.append(dice_score.detach().cpu().numpy())
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            
            ### save model    ######
            checkpoint = {
                "state_dict": model1.state_dict(),
                "optimizer":optimizer1.state_dict(),
            }
            save_checkpoint(checkpoint)
