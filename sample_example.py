'''
Small demo on how the kurwaswamy distribtution works.
'''

import torch




l = -1.1
r = 1.1

probs = torch.rand((32, 5))
a = torch.rand((32, 5))
b = torch.rand((32, 5))


# Now we randomly sample from it.

def invert_k(u, a, b):
    return (1 - (1 - u)**1/a)**1/b


k = invert_k(probs, a,b)

t = l + (r - l)*k

print(t)
maxed = t < 0


t_maxed = (t * ~maxed) + torch.zeros(t.shape) * maxed
mined = t_maxed > 1

h = (t_maxed * ~mined) + torch.ones(t_maxed.shape) * mined



print(h)
