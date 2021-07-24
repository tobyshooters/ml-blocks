import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceMeshBlock(nn.Module):
    """
    Residual block with one dw-conv and max-pool/channel pad in the second
    branch if input channels doesn't match output channels
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(FaceMeshBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.act(self.convs(h) + x)


class FaceMesh(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(16),
            FaceMeshBlock(16, 16),
            FaceMeshBlock(16, 16),
            FaceMeshBlock(16, 32, stride=2),
            FaceMeshBlock(32, 32),
            FaceMeshBlock(32, 32),
            FaceMeshBlock(32, 64, stride=2),
            FaceMeshBlock(64, 64),
            FaceMeshBlock(64, 64),
            FaceMeshBlock(64, 128, stride=2),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128, stride=2),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128),
        )
        
        self.coord_head = nn.Sequential(
            FaceMeshBlock(128, 128, stride=2),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128),
            nn.Conv2d(128, 32, 1),
            nn.PReLU(32),
            FaceMeshBlock(32, 32),
            nn.Conv2d(32, 1404, 3)
        )
        
        self.conf_head = nn.Sequential(
            FaceMeshBlock(128, 128, stride=2),
            nn.Conv2d(128, 32, 1),
            nn.PReLU(32),
            FaceMeshBlock(32, 32),
            nn.Conv2d(32, 1, 3)
        )
        self.pad = nn.ReflectionPad2d((1, 0, 1, 0))
        
    
    def forward(self, x):
        b = x.shape[0]
        x = self.pad(x)        # match TFLite padding
        x = self.backbone(x)   # (b, 128, 6, 6)
        c = self.conf_head(x)  # (b, 1, 1, 1)
        c = c.view(b, -1)      # (b, 1)
        r = self.coord_head(x) # (b, 1404, 1, 1)
        r = r.reshape(b, -1)   # (b, 1404)
        return r, c


    def initialize(self, weights, device):
        self.device = device
        self.load_state_dict(torch.load(weights))
        self.to(device)
        self.eval()


    def process(self, frame):
        assert frame.shape[:2] == (192, 192)

        x = torch.Tensor(frame).to(self.device)
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        x = x.float() / 127.5 - 1.0

        with torch.no_grad():
            mesh, probs = self.forward(x)
            
        mesh = mesh.view(-1, 3)
        mesh = mesh.detach().cpu().numpy()
        return mesh
