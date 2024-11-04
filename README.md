# Pytorch-GroupNorm

## Overview

This repository provides a custom implementation of Group Normalization, inspired by a discussion on the [PyTorch Forum](https://discuss.pytorch.org/t/reproducing-groupnorm-running-mean-var/174846). The `CustomGroupNorm` module builds on the concepts introduced in the original [Group Normalization paper](https://arxiv.org/pdf/1803.08494.pdf) and includes options for `subtractive` and `divisive` normalization, offering added flexibility over PyTorch’s native implementation.

## Files

- **groupnorm.py**: Defines the `CustomGroupNorm` class.
- **test.py**: Contains tests that validate `CustomGroupNorm` against PyTorch's native `GroupNorm` implementation.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/Pytorch-GroupNorm.git
cd Pytorch-GroupNorm
```

Ensure you have PyTorch installed:

```bash
pip install torch
```

## Usage

To use `CustomGroupNorm` in your project:

```python
from groupnorm import CustomGroupNorm
import torch.nn as nn

# Example usage
norm_layer = CustomGroupNorm(num_groups=2, num_channels=3, affine=True)
```

### Parameters

- `num_groups` (int): Number of groups to divide the channels into.
- `num_channels` (int): Number of input channels.
- `affine` (bool): If `True`, this module has learnable affine parameters.
- `eps` (float): A small value to avoid division by zero.
- `subtractive` (bool): If `True`, applies subtractive normalization.
- `divisive` (bool): If `True`, applies divisive normalization.

## Testing

You can run `test.py` to compare `CustomGroupNorm` against PyTorch’s native `GroupNorm`:

```bash
python test.py
```

This will print the maximum absolute difference between the two implementations, which should be very small if they are consistent.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.
