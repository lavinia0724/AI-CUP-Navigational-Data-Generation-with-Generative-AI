import os
import cv2
import torch
import torchvision

path = "../Training_dataset/img"

jpg_file_names = [file for file in os.listdir(path) if file.endswith('.jpg')]
jpg_file_names.sort()

for jpg_file_name in jpg_file_names:

    img = cv2.imread(path + "/" + jpg_file_name)
    img = torch.from_numpy(img)


    img = img.permute(2, 0, 1)  # img.shape [3 , 240 , 428]

    img = torchvision.transforms.functional.hflip(img)

    img = img.permute(1, 2, 0)   # img.shape [240 , 428 , 3]



    cv2.imwrite(path + "/" + jpg_file_name.replace(".jpg", "hflip.jpg") , img.cpu().numpy())



path = "../Training_dataset/label_img"

jpg_file_names = [file for file in os.listdir(path) if file.endswith('.png')]
jpg_file_names.sort()

for jpg_file_name in jpg_file_names:

    img = cv2.imread(path + "/" + jpg_file_name)
    img = torch.from_numpy(img)


    img = img.permute(2, 0, 1)  # img.shape [3 , 240 , 428]

    img = torchvision.transforms.functional.hflip(img)

    img = img.permute(1, 2, 0)   # img.shape [240 , 428 , 3]



    cv2.imwrite(path + "/" + jpg_file_name.replace(".png", "hflip.png") , img.cpu().numpy())
