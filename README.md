# Dilated-Convolution-with-Learnable-Gaussians-PyTorch

**Install the latest developing version from the source codes**:

From [GitHub](https://github.com/icml-workshop/Dilated-Convolution-with-Learnable-Gaussians-PyTorch):
```bash
git clone https://github.com/icml-workshop/Dilated-Convolution-with-Learnable-Gaussians-PyTorch.git
cd Dilated-Convolution-with-Learnable-Gaussians-PyTorch
python3 -m pip install --upgrade pip
python3 -m build 
python3 -m pip install dist/dclg-0.0.1-py3-none-any.whl 

```

## Usage
Dcls methods could be easily used as a substitue of Pytorch's nn.Conv**n**d classical convolution method:

```python
import torch
from DCLG.Conv import  Dcls2d

# With square kernels, equal stride and dilation
m = Dcls2d(16, 33, kernel_count=3, dilated_kernel_size=7, version='gauss')
input = torch.randn(20, 16, 50, 100)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad, m.SIG.grad)
```
A typical use is with the separable convolution

```python
import torch
from DCLS.construct.modules import  Dcls2d

m = Dcls2d(96, 96, kernel_count=34, dilated_kernel_size=17, padding=8, groups=96, version='gauss')
input = torch.randn(128, 96, 56, 56)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad, m.SIG.grad)
```

Dcls with different dimensions 
```python
import torch
from DCLS.construct.modules import  Dcls1d 

# Will construct kernels of size 7x7 with 3 elements inside each kernel
m = Dcls1d(3, 16, kernel_count=3, dilated_kernel_size=7, version='gauss')
input = torch.rand(8, 3, 32)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad, m.SIG.grad)
```

```python
import torch
from DCLS.construct.modules import  Dcls3d

m = Dcls3d(16, 33, kernel_count=10, dilated_kernel_size=(7,8,9), version='gauss')
input = torch.randn(20, 16, 50, 100, 30)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad, m.SIG.grad)
```
