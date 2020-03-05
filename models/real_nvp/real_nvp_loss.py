import numpy as np
import torch.nn as nn


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):

        # log ( p_X(x) ) = log( p_Z(f(x)) ) + log | det J | 
        # log( p_Z(f(x)) ) = prior_ll
        # log | det J | = sldj
        
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        
        # print('prior_ll',prior_ll.shape)
        #TODO: What is happening here?

        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])

        # print('prior_ll',prior_ll)
        
        ll = prior_ll + sldj
        # print(ll)
        nll = -ll.mean()
 
        return nll
