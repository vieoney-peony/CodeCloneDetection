import torch

input1 = torch.randn(100, 128).mean(0)
input2 = torch.randn(100, 128).mean(0)
output = torch.cosine_similarity(input1, input2)
print(output)
print(input1.shape)
