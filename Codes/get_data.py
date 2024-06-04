import os
import cv2

import numpy as np

import torch
import torchvision


# from hyper_parameter import (device,
#                              dtype,
#                              batch_size,
#                              hflip_p)

from hyper_parameter import (device,
                             dtype,
                             batch_size)


import matplotlib.pyplot as plt


def get_data():

    jpg_file_names = [file for file in os.listdir("../Training_dataset/img") if file.endswith('.jpg')]
    jpg_file_names.sort()

    idxs = np.random.choice(range(len(jpg_file_names)), size=batch_size, replace=False)

    data = None

    for i in range(batch_size):
        img = cv2.imread("../Training_dataset/img/" + jpg_file_names[idxs[i]])
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        print(img.shape)

        if data == None:
            data = img
        else:
            data = torch.cat((data , img) , dim=0)

    label = None

    for i in range(batch_size):
        img = cv2.imread("../Training_dataset/label_img/" + jpg_file_names[idxs[i]].replace("jpg" , "png"))
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        if label == None:
            label = img
        else:
            label = torch.cat((label , img) , dim=0)

    data = data.to(dtype=dtype , device=device)
    label = label.to(dtype=dtype , device=device)

    data = data / 255.
    label = label / 255.

    # data.shape  [batch_size , 240 , 428 , 3]
    # label.shape [batch_size , 240 , 428 , 3]

    data = data.permute(0, 3, 1, 2)     # data.shape  [batch_size , 3 , 240 , 428]
    label = label.permute(0, 3, 1, 2)   # label.shape [batch_size , 3 , 240 , 428]

    return data , label

    # data.shape  [batch_size , 3 , 240 , 428]
    # label.shape [batch_size , 3 , 240 , 428]





# def get_data_with_hflip():


#     jpg_file_names = [file for file in os.listdir("../Training_dataset/img") if file.endswith('.jpg')]
#     jpg_file_names.sort()

#     idxs = np.random.choice(range(len(jpg_file_names)), size=batch_size, replace=False)

#     hflip_bool = np.random.choice((True , False), size=batch_size, replace=True, p=(hflip_p , 1. - hflip_p))


#     data = None

#     for i in range(batch_size):
#         img = cv2.imread("../Training_dataset/img/" + jpg_file_names[idxs[i]])
#         img = torch.from_numpy(img)
#         img = img.permute(2, 0, 1)  # img.shape [3 , 240 , 428]


#         if hflip_bool[i] == True:
#             img = torchvision.transforms.functional.hflip(img)


#         img = img.unsqueeze(0)  # img.shape [1 , 3 , 240 , 428]


#         if data == None:
#             data = img
#         else:
#             data = torch.cat((data , img) , dim=0)

#     # data.shape [batch_size , 3 , 240 , 428]



#     label = None

#     for i in range(batch_size):
#         img = cv2.imread("../Training_dataset/label_img/" + jpg_file_names[idxs[i]].replace("jpg" , "png"))
#         img = torch.from_numpy(img)
#         img = img.permute(2, 0, 1)  # img.shape [3 , 240 , 428]


#         if hflip_bool[i] == True:
#             img = torchvision.transforms.functional.hflip(img)

#         img = img.unsqueeze(0) # img.shape [1 , 3 , 240 , 428]


#         if label == None:
#             label = img
#         else:
#             label = torch.cat((label , img) , dim=0)

#     # label.shape [batch_size , 3 , 240 , 428]



#     data = data.to(dtype=dtype , device=device)
#     label = label.to(dtype=dtype , device=device)

#     data = data / 255.
#     label = label / 255.

#     # data.shape  [batch_size , 3 , 240 , 428]
#     # label.shape [batch_size , 3 , 240 , 428]





#     return data , label

#     # data.shape  [batch_size , 3 , 240 , 428]
#     # label.shape [batch_size , 3 , 240 , 428]