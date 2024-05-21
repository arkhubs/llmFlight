# %%
import torch, sys, os, cpuinfo
sys.path.append(os.path.join(os.getcwd(), '../../'))
from embedding_models.text_embedding_3_small import get_embeddings

# %%
import torch.nn as nn       # 用于继承的模型都在这里
import torch.nn.functional as F     # 常用的激活函数、损失函数等

class LiHuNet(nn.Module):
    def __init__(self):     # 初始化
        super().__init__() # init父类
        ## 构建全连接层
        self.fc = nn.Sequential(
            nn.Linear(1536 * 2, 816), 
            nn.ReLU(),
            nn.Linear(816, 256),
            nn.ReLU(),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        ) # 注意这里只进行了线性变换，没有接softmax变换，这是因为随后的损失函数中包含了softmax操作。

    def forward(self, x, device):
        x = x.to(device)       # 前向传播
        output = self.fc(x.view(x.size()[0], -1))
        return output


# %%
class Model():
    def __init__(self, pth, device):
        model = LiHuNet()
        model.load_state_dict(torch.load(pth))
        model.to(device)
        model.eval()
        self.model = model
        self.device = device
    def infer(self, question, answer):
        return self.model.forward(torch.cat((torch.tensor(question), torch.tensor(answer)), dim=0).view(1, -1), device=self.device).item()


# %%
def supported_devices():
    devices = {}

    try:
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    except Exception:
        print('require package py-cpuinfo')
        cpu_name = 'cpu'
    devices['CPU', cpu_name] = torch.device('cpu')

    try:
        import torch_directml
        device, device_name = torch_directml.device(), torch_directml.device_name(0)
        devices['DirectML', device_name] = device
    except Exception:
        pass

    for i in range(torch.cuda.device_count()):
        devices['CUDA', torch.cuda.get_device_name(i)] = torch.device(f'cuda:{i}')
    
    return devices
