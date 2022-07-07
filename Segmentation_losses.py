#class IoULoss(nn.Module):
#    def __init__(self, weight=None, size_average=True):
#        super(IoULoss, self).__init__()
#
#    def forward(self, inputs, targets, smooth=1):
#        inputs = inputs.view(-1)
#        targets = targets.view(-1)
#        intersection = (inputs * targets).sum()
#        total = (inputs + targets).sum()
#        union = total - intersection   
#        IoU = (intersection + smooth)/(union + smooth)          
#        return 1 - IoU
        


#loss_ = torch.nn.BCEWithLogitsLoss(reduction='mean')
#class IoULoss_BCE(nn.Module):
#    def __init__(self, weight=None, size_average=True):
#        super(IoULoss_BCE, self).__init__()
#
#    def forward(self, inputs, targets, smooth=1):
#    
#        inputs1 = torch.sigmoid(inputs)  
#        inputs1 = inputs1.view(-1)
#        targets = targets.view(-1)
#        intersection = (inputs1 * targets).sum()
#        total = (inputs1 + targets).sum()
#        union = total - intersection   
#        IoU = (intersection + smooth)/(union + smooth)          
#        IoU_Loss = 1 - IoU
#        
#        inputs2 = inputs.view(-1)
#        BCE = loss_(inputs2, targets) 
#        total_loss = IoU_Loss+ BCE
#        
#        return total_loss   
        
#class BCE_loss(nn.Module):
#    def __init__(self, weight=None, size_average=True):
#        super(BCE_loss, self).__init__()
#
#    def forward(self, inputs, targets, smooth=1):
#        inputs = inputs.view(-1)
#        targets = targets.view(-1)
#          
#        BCE = loss_(inputs, targets, reduction='mean') 
#        
#        return BCE
        
             
ALPHA = 0.3
BETA = 0.7
GAMMA = .75

class FocalTverskyLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self)._init_()

    def forward(self, inputs, targets, smooth=.0001, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
