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

# for boundaries ###
def mask_to_boundary(mask):

    h, w = mask.shape
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=5)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode
class IoULoss_b(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss_b, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs=mask_to_boundary(inputs)
        targets=mask_to_boundary(targets)
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection   
        IoU = (intersection + smooth)/(union + smooth)          
        return 1 - IoU


## Dice Loss Multi-Class ##

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))
