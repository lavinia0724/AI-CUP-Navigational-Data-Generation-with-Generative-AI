import torch

##### 運行設備 #####
device = "cuda"
# device = "cpu"


##### 運算精度 #####
dtype = torch.float32


##### 訓練超參 #####
max_epoch = 8000

batch_size = 32

learning_rate = 1e-3


##### 損失函數 #####
loss_f = "bce"
# loss_f = "mse"









##### 資料是否水平翻轉 #####
# hflip_bool = True
# hflip_bool = False
# hflip_p = 0.3   # 0 ~ 1



##### 是否進行黑白平衡，是否對黑色補強 #####
# balance_bool = True
# balance_bool = False

# balance = 1.1   # origin balance = 1.1



