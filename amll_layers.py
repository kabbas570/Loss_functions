class Move_Axis(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.source,self.destination = args
    def forward(self, x):
         x = torch.moveaxis(x,(self.source),(self.destination))  ## --->  [b,c,d,h,w] --> [b,c,h,w,d] 
         return x

class SeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.squeeze(x, 4)
         return x   

class UnSeQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
         x = torch.unsqueeze(x, 4)
         return x 
             
class flat(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = nn.Flatten(3,4)
    def forward(self, x):
         x = self.shape(x)
         return x 

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
