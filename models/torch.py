from torch.nn import Module


class Identity(Module):
    def forward(self, x):
        return x
