from groupnorm import CustomGroupNorm
import torch
import torch.nn as nn

x = torch.randn(32, 3, 28, 28) # batch_size, num_channels, height, width
norm1 = nn.GroupNorm(1, 3, affine=False, eps=1e-5)
norm2 = CustomGroupNorm(1, 3, affine=False, eps=1e-5)
y1 = norm1(x)[3][0][0][1]
y2 = norm2(x)[3][0][0][1]
print(y1, y2)
print((norm1(x) - norm2(x)).abs().max() < 10e-7)
