import torch



a = torch.tensor([1,0,0,1]).float()






b = torch.randn(2,4, 1).float()
b = b.squeeze(dim=-1)
a = a.repeat(2, 1)
print(a)
print(b.shape)
print(a.shape)
print(b)
x = (a * b).sum(dim=0)
print(x)
print(x.shape)


