import os, sys
import zipfile
import tarfile

# 要压缩的文件夹路径
root = sys.path[0]
source_folder = os.path.join(root, "../../")

# 要排除的相对路径列表
exclude_paths = [
    ".git",
    "dataset/history",
    "dataset/origin",
    "dataset/test_gpt_0125.csv",
    "dataset/test_human.csv",
    "dataset/test_questions.csv",
    "main_models/LiHuNet3072-v1-turbo/trainlog",
    "releases"
]

# 创建相对路径的排除函数
def should_exclude(path):
    for exclude_path in exclude_paths:
        if os.path.commonpath([os.path.join(source_folder, exclude_path)]) == os.path.commonpath([path, os.path.join(source_folder, exclude_path)]):
            return True
    return False

# 压缩为.zip文件
zip_filename = "../llmFlight.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_folder):
        if should_exclude(root):
            print("\033[91m", root, "excluded\033[0m")
            continue
        for file in files:
            file_path = os.path.join(root, file)
            if should_exclude(file_path):
                print("\033[91m", file_path, "excluded\033[0m")
                continue
            zipf.write(file_path, os.path.relpath(file_path, source_folder))
            print("\033[92m", file_path, "compressed\033[0m")

# 压缩为.tgz文件
tgz_filename = "../llmFlight.tgz"
with tarfile.open(tgz_filename, "w:gz") as tar:
    for root, dirs, files in os.walk(source_folder):
        if should_exclude(root):
            print("\033[91m", root, "excluded\033[0m")
            continue
        for file in files:
            file_path = os.path.join(root, file)
            if should_exclude(file_path):
                print("\033[91m", file_path, "excluded\033[0m")
                continue
            tar.add(file_path, arcname=os.path.relpath(file_path, source_folder))
            print("\033[92m", file_path, "compressed\033[0m")
