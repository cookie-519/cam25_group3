import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Tanh()
        )


    def forward(self, x):
        return self.main(x)