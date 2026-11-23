
import torch

print("PyTorch Installation Successful!")

print("PyTorch Version:", torch.__version__)

print("CUDA Available:", torch.cuda.is_available())

# Create a tensor
a = torch.tensor([10, 20, 30])
b = torch.tensor([1, 2, 3])

result = a + b

print("Tensor Addition Result:", result)

print("Processing Device: CPU")
