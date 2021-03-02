import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxBlurPool(nn.Module):
    """
    Based on: https://richzhang.github.io/antialiased-cnns/
    Max:  performed densely, i.e. stride 1
    Pad:  reflect the border to preserve edge values
    Blur: each channel independently with Gaussian filter
    Pool: done via the stride 2 in the blur
    """
    def __init__(self, n_channels):
        super(MaxBlurPool, self).__init__()
        self.max = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pad = nn.ReflectionPad2d(1)

        a = np.array([1., 2., 1.])                            # 1d filter
        f = torch.FloatTensor(a[None, :] * a[:, None])        # outer-product
        f /= torch.sum(f)                                     # normalize
        f = f[None, None, :, :].repeat((n_channels, 1, 1, 1)) # channels deep
        self.register_buffer('f', f)                          # move to CUDA

    def forward(self, x):
        x = self.max(x)
        x = self.pad(x)
        x = F.conv2d(x, self.f, stride=2, groups=x.shape[1])
        return x
