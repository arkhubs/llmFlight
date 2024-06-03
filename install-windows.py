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

import os
import subprocess
import requests
from zipfile import ZipFile

def download_and_extract_zip(url, extract_to):
    # 下载ZIP文件
    local_zip_path = os.path.join(extract_to, "downloaded.zip")
    response = requests.get(url)
    with open(local_zip_path, 'wb') as f:
        f.write(response.content)

    # 解压ZIP文件
    with ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # 删除下载的ZIP文件
    os.remove(local_zip_path)


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
            "installing_python": "正在安装python3.10 ...",
            "installing_pip": "正在安装pip ...",
            "rewriting_pth": "正在重写python310._pth ...",
            "choose_option": "请选择安装选项：",
            "option1": "[1] 安装CPU版本PyTorch，附带DirectML（默认）",
            "option2": "[2] 仅安装CPU版本PyTorch",
            "option3": "[3] 安装CUDA 11.8版本PyTorch",
            "option4": "[4] 安装CUDA 12.1版本PyTorch",
            "invalid_option": "无效选项，请重新运行程序并选择有效的选项。",
            "install_success": "安装成功！",
            "start_now": "是否立即启动？[YES/NO]：",
            "completed": "安装程序完成。您可以随时手动启动应用程序。",
            "directml": "是否附带DirectML？[YES/NO]："
        }
    else:
        messages = {
            "start_installation": "Press any key to start installation...",
            "installing_python": "Installing Python 3.10...",
            "installing_pip": "Installing pip...",
            "rewriting_pth": "Rewriting python310._pth...",
            "choose_option": "Please choose an installation option:",
            "option1": "[1] Install CPU version PyTorch with DirectML (default)",
            "option2": "[2] Install CPU version PyTorch only",
            "option3": "[3] Install PyTorch with CUDA 11.8",
            "option4": "[4] Install PyTorch with CUDA 12.1",
            "invalid_option": "Invalid option, please rerun the program and choose a valid option.",
            "install_success": "Installation successful!",
            "start_now": "Would you like to start now? [YES/NO]:",
            "completed": "Installation complete. You can start the application manually anytime.",
            "directml": "Include DirectML? [YES/NO]:"
        }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    input(messages["start_installation"])

    # 安装python3.10
    # https://www.python.org/ftp/python/3.10.10/python-3.10.10-embed-amd64.zip
    # https://www.python.org/ftp/python/3.10.10/Python-3.10.10.tgz  
    print(messages["installing_python"])
    zip_url = "https://www.python.org/ftp/python/3.10.10/python-3.10.10-embed-amd64.zip"
    extract_directory = "./Scripts/python310"
    # 确保解压目录存在
    os.makedirs(extract_directory, exist_ok=True)
    download_and_extract_zip(zip_url, extract_directory)
    print("Python 3.10 installed successfully!")

    # 安装pip
    print(messages["installing_pip"])
    getpip = requests.get("https://bootstrap.pypa.io/get-pip.py")
    with open("./Scripts/python310/get-pip.py", 'wb') as f:
        f.write(getpip.content)
    subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "./Scripts/python310/get-pip.py"])
    print("Pip installed successfully!")

    # 重写python310._pth
    print(messages["rewriting_pth"])
    with open("./Scripts/python310/python310._pth", 'w', encoding='utf-8') as file:
        file.write('''
python310.zip
.

# Uncomment to run site.main() automatically
import site
        ''')
    print("python310._pth rewritten successfully!")

    # 安装PyTorch
    print(messages["choose_option"])
    print(messages["option1"])
    print(messages["option2"])
    print(messages["option3"])
    print(messages["option4"])

    choice = input("输入选项编号并按下回车 / Enter choice number and press Enter: ")

    if choice == '1' or choice == '':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--no-warn-script-location"])
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch-directml==0.2.1.dev240521", "--no-warn-script-location"])
    elif choice == '2':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--no-warn-script-location"])
    elif choice == '3':
        dml_choice = input(messages["directml"])
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--index-url", "https://download.pytorch.org/whl/cu118", "--no-warn-script-location"])
        if dml_choice.upper() == 'YES':
            subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch-directml==0.2.1.dev240521", "--no-warn-script-location"])
    elif choice == '4':
        dml_choice = input(messages["directml"])
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--no-warn-script-location"])
        if dml_choice.upper() == 'YES':
            subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "torch-directml==0.2.1.dev240521", "--no-warn-script-location"])
    else:
        print(messages["invalid_option"])
        return

    subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "pip", "install", "-r", os.path.join(script_dir, "requirements.txt"), "--no-warn-script-location"])

    print(messages["install_success"])
    launch_choice = input(messages["start_now"])

    if launch_choice.upper() == 'YES':
        subprocess.run([os.path.join(script_dir, "Scripts", "python310", "python"), "-m", "streamlit", "run", os.path.join(script_dir, "webui", "AIContentDetector.py"), "--server.port", "8502"])
    else:
        print(messages["completed"])

if __name__ == "__main__":
    main()