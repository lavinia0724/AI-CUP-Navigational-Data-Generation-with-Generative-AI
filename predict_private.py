import os
import torch
import cv2
import matplotlib.pyplot as plt

from hyper_parameter import dtype , device


folder = "upload1"
model_path = "model1.pt"


if not os.path.isdir(folder):
    os.mkdir(folder)



model = torch.load(model_path)
model = model.to(device)
model = model.eval()


private_file_names = [file for file in os.listdir("../Private Test Image") if file.endswith('.jpg')]
private_file_names.sort()


for private_file_name in private_file_names:


    data = cv2.imread("../Private Test Image/" + private_file_name)
    data = torch.from_numpy(data)
    data = data.to(dtype=dtype)
    data = data / 255
    data = data.to(device=device)

    data = data.unsqueeze(0)
    data = data.permute(0, 3, 1, 2)


    with torch.no_grad():

        predict = model(data)


        predict = predict.permute(0, 2, 3, 1)
        predict = predict.squeeze(0)


        # plt.imshow(predict.cpu().numpy())
        # plt.pause(100)


    cv2.imwrite(folder + "/" + private_file_name.replace("jpg","png") , predict.cpu().numpy() * 255)


