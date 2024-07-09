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

import os, sys
import subprocess
import requests


def get_language_choice():
    print("请选择语言 / Please select language:")
    print("[1] 中文")
    print("[2] English")
    choice = input("输入选项编号并按下回车 / Enter choice number and press Enter: ")
    return choice

def main():
    language_choice = get_language_choice()

    if language_choice == '1':
        messages = {
            "start_installation": "按下任意键开始安装...",
            "installing_python": "正在编译安装python3.10 ...",
            "choose_option": "请选择安装选项：",
            "option2": "[1] 仅安装CPU版本PyTorch",
            "option3": "[2] 安装CUDA 11.7版本PyTorch",
            "option4": "[3] 安装CUDA 11.8版本PyTorch",
            'option5': "[4] 安装CUDA 12.1版本Pytorch",
            "invalid_option": "无效选项，请重新运行程序并选择有效的选项。",
            "install_success": "安装成功！",
            "start_now": "是否立即启动？[YES/NO]：",
            "completed": "安装程序完成。您可以随时手动启动应用程序。",
        }
    else:
        messages = {
            "start_installation": "Press any key to start installation...",
            "installing_python": "Compiling and installing Python 3.10...",
            "choose_option": "Please choose an installation option:",
            "option2": "[1] Install CPU version PyTorch only",
            "option3": "[2] Install PyTorch wirh CUDA 11.7",
            "option4": "[3] Install PyTorch wirh CUDA 11.8",
            'option5': "[4] Install PyTorch wirh CUDA 12.1",
            "invalid_option": "Invalid option, please rerun the program and choose a valid option.",
            "install_success": "Installation successful!",
            "start_now": "Would you like to start now? [YES/NO]:",
            "completed": "Installation complete. You can start the application manually anytime.",
        }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    input(messages["start_installation"])

    # 安装python3.10
    # https://www.python.org/ftp/python/3.10.10/python-3.10.10-embed-amd64.zip
    # https://www.python.org/ftp/python/3.10.10/Python-3.10.10.tgz  
    print(messages["installing_python"])
    tar_url = "https://www.python.org/ftp/python/3.10.10/Python-3.10.10.tgz"
    extract_directory = os.path.join(script_dir, "Scripts/python310")
    # 确保解压目录存在
    os.makedirs(extract_directory, exist_ok=True)
    subprocess.run(["wget", tar_url], shell=True)
    subprocess.run(["tar", "-zxvf", "Python-3.10.10.tgz", "--strip-components 1", "-C", extract_directory], shell=True)
    subprocess.run(["rm", "-f", "Python-3.10.10.tgz"])
    subprocess.run(["cd", extract_directory])
    subprocess.run(["./configure", "--enable-optimizations", f"--prefix={extract_directory}"])
    subprocess.run(["make", "-j", "$(nproc)"])
    subprocess.run(["make", "install"])
    print("Python 3.10 installed successfully!")

    # 安装PyTorch
    print(messages["choose_option"])
    print(messages["option1"])
    print(messages["option2"])
    print(messages["option3"])
    print(messages["option4"])

    choice = input("输入选项编号并按下回车 / Enter choice number and press Enter: ")

    if choice == '1':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--no-warn-script-location"])
    elif choice == '2':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.1", "torchaudio==2.0.1", "--index-url", "https://download.pytorch.org/whl/cu117", "--no-warn-script-location"])
    elif choice == '3':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--index-url", "https://download.pytorch.org/whl/cu118", "--no-warn-script-location"])
    elif choice == '4':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--no-warn-script-location"])
    else:
        print(messages["invalid_option"])
        return

    subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "-r", os.path.join(script_dir, "requirements.txt"), "--no-warn-script-location"])
    subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "ipykernel", "install", "--user", "--name=lf"  ])

    print(messages["install_success"])
    launch_choice = input(messages["start_now"])

    if launch_choice.upper() == 'YES':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "streamlit", "run", os.path.join(script_dir, "webui", "LSR.py"), "--server.port", "8502"])
    else:
        print(messages["completed"])

if __name__ == "__main__":
    main()