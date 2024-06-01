import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from hyper_parameter import (batch_size,
                             loss_f)




def loss_function(predict, label):


    if loss_f == "bce":
        loss = nn.BCELoss(reduction='sum')(predict, label)
    elif loss_f == "mse":
        loss = ((label - predict) ** 2).sum()
    else:
        print("wrong loss function")


    loss = loss / batch_size


    return loss







# def loss_function(predict, label):

#     n, c, h, w = predict.size()
#     weights = np.ones((n, c, h, w))

#     if balance_bool:
#         for i in range(n):
#             t = label[i, :, :, :].cpu().data.numpy()
#             pos = (t == 1).sum()
#             neg = (t == 0).sum()
#             valid = neg + pos
#             weights[i, t == 1] = neg * 1. / valid
#             weights[i, t == 0] = pos * balance / valid

#     weights = torch.Tensor(weights)
#     weights = weights.to(device=device)


#     if loss_f == "bce":
#         loss = nn.BCELoss(weights, reduction='sum')(predict, label)
#     elif loss_f == "mse":
#         loss = (((label - predict) ** 2) * weights).sum()
#     else:
#         print("wrong loss function")


#     loss = loss / batch_size


#     return loss

