## From https://github.com/MinaGhadimiAtigh/Hyperbolic-Busemann-Learning/blob/master/prototype_learning.py

import torch
import math 

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# When prototype dimension >=3, the goal is to find the largest cosine similarity between pairs of prototypes and minimize it.
def prototype_loss(prototype):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototype, prototype.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]

    return loss.mean(), product.max()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# When prototype dimension=2, learning can be easily performed by splitting the unit-cirlce into equal parts, 
# seperated by an angle of 2π/n, which n stands for the number of classes. 
# Then, for each angle ψ, the coordinates are obtained as (cos ψ,sin ψ).

def prototype_unify(num_classes):
    single_angle = 2 * math.pi / num_classes
    help_list = np.array(range(0, num_classes))
    angles = (help_list * single_angle).reshape(-1, 1)

    sin_points = np.sin(angles)
    cos_points = np.cos(angles)

    set_prototypes = torch.tensor(np.concatenate((cos_points, sin_points), axis=1))
    return set_prototypes

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_prototypes(dims, num_classes, epochs=1000, learning_rate=0.1, momentum=0.9):
    # Now, prototype learning can be performed. 
    # While for d=2 prototype learning consists of only splitting the unit-circle, for d>2, training and optimization is needed.
    if dims == 2:
        prototypes = prototype_unify(num_classes)
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))

    elif dims > 2:
        prototypes = torch.randn(num_classes, dims)
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
        optimizer = optim.SGD([prototypes], lr=learning_rate, momentum=momentum)
        # Optimize for separation.
        for i in range(epochs):
            # Compute loss.
            loss, _ = prototype_loss(prototypes)
            # Update.
            loss.backward()
            optimizer.step()
            # Normalize prototypes again
            prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
            optimizer = optim.SGD([prototypes], lr=learning_rate, momentum=momentum)
            
    return prototypes
        
