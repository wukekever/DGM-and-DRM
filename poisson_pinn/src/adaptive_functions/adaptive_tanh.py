import torch
from torch.nn.parameter import Parameter

class AdaptiveTanh(torch.nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(AdaptiveTanh, self).__init__()
        #self.in_features = in_features

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(1.0)) # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
            
        self.alpha.requiresGrad = True # set requiresGrad to true!

        self.scale = Parameter(torch.tensor(1.0))
        self.scale.requiresGrad = True # set requiresGrad to true!

        self.translate = Parameter(torch.tensor(0.0))
        self.translate.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        x += self.translate
        return self.scale * (torch.exp(self.alpha * x) - torch.exp(-self.alpha * x))/(torch.exp(self.alpha * x) + torch.exp(-self.alpha * x))
