"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/2/28 16:47
@Software: PyCharm 
@File : Format_conversion.py
"""

import os
from PIL import Image
from tqdm import tqdm

path = '../data'  # 图片所在目录路径
exts = ['.gif', '.jpg', '.webp', '.png']   # 需要重命名的文件扩展名
new_ext = '.png'  # 新的文件扩展名

# 获取目录下所有需要重命名的文件
files = [f for f in os.listdir(path) if os.path.splitext(f)[-1] in exts]

# 重命名文件并保存为.png格式
for i, f in tqdm(enumerate(files), total=len(files)):
    old_path = os.path.join(path, f)
    new_path = os.path.join(path, str(i+1) + new_ext)
    if os.path.splitext(f)[-1] in ['.gif', '.webp']:
        os.remove(old_path)
        continue
    else:
        with Image.open(old_path) as img:
            img = img.convert('RGB')  # 转换为RGB格式
            img.save(new_path)
            os.system("convert "+new_path+" "+new_path)  # sudo apt install imagemagick
        os.remove(old_path)

"""
文件结构：
awards/
├── 1.png
├── 2.png
├── 3.png
"""
