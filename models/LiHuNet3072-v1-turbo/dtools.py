'''
Copyright 2024 Zhixuan Hu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

## version 0.24.5.0
## 0.24.5.0 update loc 2 -> 4

import torch
import IPython.display as display                   # 用于控制notebook的输出流
import io, os, re, glob, time, json                 # 一些io库
import collections, types, traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                     # 绘图工具包
from threading import Thread                        # 多线程，用于减少I/O操作对训练进程的阻塞

def select_device(device=''):       # 若指定device则手动选择，否则自动选择
    s = " selected."
    if device != '':
        device, device_name = device, str(device)
    elif torch.cuda.is_available():
        device, device_name = torch.device('cuda'), 'cuda'
    else:
        try:
            import torch_directml
            device, device_name = torch_directml.device(), torch_directml.device_name(0)
        except Exception:
            device, device_name = torch.device('cpu'), 'cpu'
    print(device_name + s)
    return device, device_name

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

class Controller():

    def __init__(self):
        self.registry = {}
    
    def run(self, ops):
        if isinstance(ops, list):
            for op in ops:
                if isinstance(op, tuple):
                    opt = Thread(target=op[0], args=op[1])
                else:
                    opt = Thread(target=self.run, args=(op,))
                opt.start(); opt.join()
        elif isinstance(ops, dict):
            opts = []
            for op in ops.values():
                if isinstance(op, tuple):
                    opt = Thread(target=op[0], args=op[1])
                else:
                    opt = Thread(target=self.run, args=(op,))
                opts.append(opt)
            for opt in opts:
                opt.start()
            for opt in opts:
                opt.join()

    def oprs(self, name, **kwargs):
        ops, fixed_args = self.registry[name]
        ops = ops(**fixed_args, **kwargs)
        return ops

    def register(self, name:str, ops:types.FunctionType, **fixed_args):
        self.registry[name] = (ops, fixed_args)

# 定义日志记录器
class Marker():

    def __init__(self, makedir: bool=True, workdir: str=os.getcwd()+'/', epochs: int=3, 
                 columns: list=None, **hyperara): 
        self.workdir = workdir                      # 工作目录
        self.logs = ""
        self.epoch = 0                              # 初始化epoch
        self.epochs = epochs                        # 初始化epoch上限
        self.old_epoch = 0
        self.from_epoch = 0
        self.maxmin_value = {}
        self.begin_time = time.time()               # 记录程序开始的时间
        self.begin_timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(self.begin_time))
        hyperara.update({"epochs": self.epochs})      
        self.hyperpara = hyperara                     # kwargs是记录在logs.json的超参数
        if makedir: os.makedirs(os.path.join(self.workdir, self.begin_timestr))
        self.columns = columns
        if self.columns is None:
            self.columns = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'pre_time', 'train_time', 'aft_time']
        self.columns += ['time_str']
        self.df = pd.DataFrame(np.empty((0,len(self.columns))), columns=self.columns)

    def stdIO(self, mark_value: dict={}, state_dict: dict={}, 
                device_name: str='', env_info: str='',
                xlabel: str='epoch', maxmin_columns: list=[('val_loss', -1), ('val_accuracy', 1)], 
                y_axes: list=[('loss', (0, 1.01, 0.05)), ('accuracy', (0, 1.01, 0.05))], 
                plots: dict={'loss': [('train_loss', 'red'), ('val_loss', 'blue')], 'accuracy': (('val_accuracy', 'orange'), ('train_accuracy', 'green'))}):
        return {1:[
                (self.mark, (mark_value,)), 
                (self.maxmin, (maxmin_columns,)),
                (self.add_log, ()),
                {
                    1:(self.cret_check, (state_dict,)),
                    2:(self.cret_json, (device_name, env_info, dict(xlabel=xlabel, y_axes=y_axes, plots=plots))),
                    3:(self.render, (xlabel, y_axes, plots)),
                    4:(self.clear, ())
                },
                (self.print_log, ()),
                (self.print_svg, ()),
            ]}

    # 训练结束
    def stdEND(self, device_name: str='', env_info: str='',
                xlabel: str='epoch', maxmin_columns: list=[('val_loss', -1), ('val_accuracy', 1)], 
                y_axes: list=[('loss', (0, 1.01, 0.05)), ('accuracy', (0, 1.01, 0.05))], 
                plots: dict={'loss': [('train_loss', 'red'), ('val_loss', 'blue')], 'accuracy': (('val_accuracy', 'orange'), ('train_accuracy', 'green'))}):
        return [
            (self.add_log, (f"Training has done using {self.df['pre_time'][self.from_epoch:].sum():.2f}s-{self.df['train_time'][self.from_epoch:].sum():.2f}s-{self.df['aft_time'][self.from_epoch:].sum():.2f}s.\nmin of validate_loss is [Epoch {self.maxmin_value['min_val_loss'][0]}] {self.maxmin_value['min_val_loss'][1]}; max of validate_accuracy is [Epoch {self.maxmin_value['max_val_accuracy'][0]}] {self.maxmin_value['max_val_accuracy'][1]}.\n", )),
            (self.clear, ()),
            (self.print_log, ()),
            (self.cret_json, (device_name, env_info, dict(xlabel=xlabel, y_axes=y_axes, plots=plots))),
            (self.print_svg, ()),
            (os.rename, (os.path.join(self.workdir, self.begin_timestr, 'logs.json'), os.path.join(self.workdir, self.begin_timestr, f"{self.df['time_str'].iloc[-1]}-logs.json"))),
            (os.rename, (os.path.join(self.workdir, self.begin_timestr, 'epochs.svg'), os.path.join(self.workdir, self.begin_timestr, f"{self.df['time_str'].iloc[-1]}-epochs.svg"))),
        ]
    
    def stdSTART(self, 
                 xlabel: str='epoch', maxmin_columns: list=[('val_loss', -1), ('val_accuracy', 1)], 
                y_axes: list=[('loss', (0, 1.01, 0.05)), ('accuracy', (0, 1.01, 0.05))], 
                plots: dict={'loss': [('train_loss', 'red'), ('val_loss', 'blue')], 'accuracy': (('val_accuracy', 'orange'), ('train_accuracy', 'green'))}):
        return [
            (self.add_log, ("Training has begun!\n",)),
            (self.render, (xlabel, y_axes, plots)),
            (self.print_log, ()),
            (self.print_svg, ()),
        ]

    # 刷新单元格输出
    def clear(self):      
        display.clear_output()
    
    def add_log(self, s=""):
        epoch = self.epoch
        df = self.df
        if s == "":
            self.logs += f"Epoch [{self.epoch}/{self.epochs}], time: {df['pre_time'].iloc[-1]:.2f}s-{df['train_time'].iloc[-1]:.2f}s-{df['aft_time'].iloc[-1]:.2f}s, loss: {df['train_loss'].iloc[-1]:.4f}, accuracy: {df['train_accuracy'].iloc[-1]:.4f}, validate_loss: {df['val_loss'].iloc[-1]:.4f}, validate_accuracy: {df['val_accuracy'].iloc[-1]:.4f}\n"
        else:
            self.logs += s

    def print_log(self):
        print(self.logs)             # 打印当前日志
    
    def print_svg(self):
        display.display(display.SVG(os.path.join(self.workdir, self.begin_timestr, 'epochs.svg')))

    def mark(self, mark_value: dict):     
        epochs = self.epochs    # epoch上限
        self.epoch += 1         # 更新epoch
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        mark_value.update({"time_str": time_str})
        self.df.loc[self.epoch] = pd.Series(mark_value)

    def maxmin(self, columns: list=[('val_loss', -1), ('val_accuracy', 1)]):
        ans = {}
        for column, k in columns:
            col_maxmin = (k*self.df[column]).idxmax()
            s = 'max_' if k==1 else 'min_'
            ans[s + column] = (int(col_maxmin), float(self.df[column][col_maxmin]))
        self.maxmin_value = ans

    # 渲染epochs.svg
    def render(self, xlabel: str='epoch', 
                y_axes: list=[('loss', (0, 1.01, 0.05)), ('accuracy', (0, 1.01, 0.05))], 
                plots: dict={'loss': [('train_loss', 'red'), ('val_loss', 'blue')], 'accuracy': (('val_accuracy', 'orange'), ('train_accuracy', 'green'))}):
        epochs = self.epochs      
        fig = plt.figure(figsize=(8, 6))            # 初始化训练曲线和验证曲线图
        axes = {y_axes[0][0]: fig.add_subplot(111)}                 # 创建子图（loss）
        for i in range(len(y_axes)):
            if i > 0: axes[y_axes[i][0]] = axes[y_axes[0][0]].twinx()
        axes[y_axes[0][0]].set_xlabel(xlabel)
        axes[y_axes[0][0]].set_xlim((1, epochs))                   # 横坐标范围
        axes[y_axes[0][0]].set_xticks(np.append(np.arange(1, epochs+1, int(epochs/10 + 0.9)), epochs)) # 横坐标刻度
        for ax, ylim in y_axes:
            axes[ax].set_ylim(ylim[0:2]) # 纵坐标范围
            axes[ax].set_yticks(np.arange(ylim[0], ylim[1], ylim[2]))# 纵坐标刻度   
            axes[ax].minorticks_on()                         # 显示小刻度
            axes[ax].set_ylabel(ax)                      # 纵轴标签   
        lines = []
        names = []
        for axname in plots.keys():
            ax = axes[axname]
            for plot, color in plots[axname]: 
                names.append(plot)    
                try:
                    x = range(1, self.epoch+1)                  # 曲线的自变量
                    lines.append(ax.plot(x, self.df[plot], color=color, linestyle='-')[0])    # train_loss曲线
                except:     # 画空白图
                    lines.append(ax.plot(np.zeros((epochs,)), color=color, linestyle='-')[0])       # 初始化train_loss曲线
        plt.legend(lines, names, loc=4, prop={'size':8})        # 合并图例
        plt.savefig(os.path.join(self.workdir, self.begin_timestr, 'epochs.svg'), format='svg', bbox_inches='tight')
        plt.close()

    def cret_json(self, device_name, env_info, dicts={}):    
        base = {
                "device": device_name,                              # 所用device
                "env_info": env_info,                               # 硬件、操作系统、pytorch版本等env信息
                "hyperparameter": self.hyperpara,                   # 自定义的超参数
                "epoch": self.epoch,                                                    # 当前epoch顺序
                "log": self.logs,                                   # 截至目前的日志
                "begin_timestr": self.begin_timestr
                }
        df_json = json.loads(self.df.to_json(orient='columns'))
        with open(os.path.join(self.workdir, self.begin_timestr, "logs.json"), "w", encoding='utf-8') as f:
            json.dump({**base, **df_json, **self.maxmin_value, **dicts}, f, indent=2, sort_keys=True, ensure_ascii=False)     # 写为多行json，方便阅读

    def device_tensors(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, collections.OrderedDict):
            return collections.OrderedDict((key, self.device_tensors(value, device)) for key, value in obj.items())
        elif isinstance(obj, dict):
            return {key: self.device_tensors(value, device) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.device_tensors(item, device) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.device_tensors(item, device) for item in obj)
        else:
            return obj

    def cret_check(self, state_dict):
        torch.save({**{
            "epochs": self.epochs,
            "epoch": self.epoch,
        }, **self.device_tensors(state_dict, 'cpu')}, os.path.join(self.workdir, self.begin_timestr, f"{self.df['time_str'].iloc[-1]}-Epoch-{self.epoch}.tar"))

    # 搜索特定检查点
    def seek_check(self, epoch, from_dir=""):
        if from_dir == "": 
            from_dir = os.path.join(self.workdir, self.begin_timestr)
        pattern = re.compile(r".*logs.*\.json")
        logs = [log for log in os.listdir(from_dir) if pattern.match(log)]
        times = []
        for log in logs:
            with open(os.path.join(from_dir, log), "r", encoding='utf-8') as f:
                info = json.load(f)
            time_strs = info["time_str"]
            if len(time_strs) >= epoch:
                times.append((log, time_strs[str(epoch)]))
        files = []
        for log, time_str in times:
            pattern = re.compile(rf".*{time_str}.*Epoch-{epoch}.*\.tar")
            for file in os.listdir(from_dir):
                if pattern.match(file):
                    files.append((log, file))
        index = 0
        if len(files) > 1: 
            choice = f"which one (1 to {len(files)}):  \n"; i = 1
            for log, file in files:
                choice += f"[{i}] {log} {file}  \n"; i += 1
            index = int(input(choice)) - 1
        elif len(files) < 1:
            raise Exception(f"Failed to find Epoch [{epoch}]")
        return (os.path.join(from_dir, files[index][0]), os.path.join(from_dir, files[index][1]))
    
    
    # 继续训练
    def continu(self, from_epoch=None, from_dir="", newdir=True, workdir="", epochs=0, 
                columns: list=None, **hyperpara):
        '''
        仅当new_dir为True时，手动修改workdir才生效。workdir默认为from_dir的上一级目录。
        '''
        if columns is None:
            if self.columns is None:
                self.columns = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'pre_time', 'train_time', 'aft_time']
                self.columns += ['time_str']
        else:
            self.columns = columns
            self.columns += ['time_str']

        if from_dir == "":
            from_dir = os.path.join(self.workdir, self.begin_timestr)

        log_path, tar_path = self.seek_check(from_epoch, from_dir=from_dir)
        checkpoint = torch.load(tar_path)
        with open(f"{log_path}", "r", encoding='utf-8') as f:
            info = json.load(f)

        if newdir:
            self.begin_timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            if workdir == "":
                self.workdir = os.path.join(from_dir, "../")
            else:
                self.workdir = workdir
            os.makedirs(os.path.join(self.workdir, self.begin_timestr))
        else:
            self.workdir = os.path.join(from_dir, "../")
            self.begin_timestr = os.path.relpath(from_dir, self.workdir)

        self.epoch = info["epoch"]
        if from_epoch != None: 
            self.epoch = from_epoch
        else:
            from_epoch = self.epoch
        self.from_epoch = from_epoch
        self.epochs = self.from_epoch + epochs

        self.logs = info["log"]

        self.hyperpara = info["hyperparameter"]                     # kwargs是记录在logs.json的超参数
        self.hyperpara.update(hyperpara)
        self.hyperpara["epochs"] = self.epochs
        
        info_columns = info.keys()
        self.df = pd.DataFrame(np.empty((0,len(self.columns))), columns=self.columns)
        for column in self.columns:
            if column in info_columns:
                self.df[column] = pd.Series(info[column])
        self.df.index = self.df.index.astype(int)
        self.df = self.df.loc[1:self.from_epoch+1, :]
        self.begin_time = time.time()              # 记录程序开始的时间 
        return checkpoint, info