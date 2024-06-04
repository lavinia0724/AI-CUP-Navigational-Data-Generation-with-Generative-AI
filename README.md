# AI CUP Navigational Data Generation with Generative AI
[AI CUP 2024] 以生成式AI建構無人機於自然環境偵察時所需之導航資訊競賽 II － 導航資料生成競賽 TEAM_5137

組員：吳建中、吳家萱、許中萁

```
├─Codes
│  └─__pycache__
├─Private Test Image
├─Public Test Image
└─Training_dataset
    ├─img
    └─label_img
```

## Environment
作業系統：Windows11<br>
程式語言：Python 3.10.4<br>
程式透過 CUDA 11.8 使用 GPU 訓練模型

其他需安裝的 Python Package 如下:
```
pip install numpy
pip install opencv-python
pip install torchvision
pip install timm
pip install matplotlib
```

## Dataset
來源：[AI CUP 2024 以生成式AI建構無人機於自然環境偵察時所需之導航資訊競賽 II － 導航資料生成競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/35)

## Training
跑訓練模型
```
python main.py
```

## Predict
使用 training 訓練完的 `model2.pt` 模型<br>
預測 public test dataset
```
python predict_public.py
```

預測 private test dataset
```
python predict_private.py
```
