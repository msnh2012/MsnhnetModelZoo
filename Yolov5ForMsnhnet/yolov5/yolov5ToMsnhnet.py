from PytorchToMsnhnet import *
Msnhnet.Export = True
from models.experimental import attempt_load
import torch

weights     = "weights/best.pt" # pt文件
msnhnetPath = "yolov5m.msnhnet" # 导出.msnhnet文件
msnhbinPath = "yolov5m.msnhbin" # 导出.msnhbin文件

model = attempt_load(weights, "cpu") 
model.eval() # cpu模式，推理模式

img = torch.rand(512*512*3).reshape(1,3,512,512) #生成随机推理数据
 
trans(model,img,msnhnetPath,msnhbinPath) #模型转换