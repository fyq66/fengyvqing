import os
import shutil
import random


def split_dataset(source_dir, train_dir, val_dir, val_ratio=0.2):#验证集比例为0.2
    # 确保目标目录存在
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 遍历每个类别的子文件夹
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 创建对应的训练和验证目录
        train_category_path = os.path.join(train_dir, category)
        val_category_path = os.path.join(val_dir, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(val_category_path, exist_ok=True)

        # 获取所有图像文件并随机打乱
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(images)

        # 计算验证集大小
        val_size = int(len(images) * val_ratio)

        # 分配图像到训练集和验证集
        val_images = images[:val_size]
        train_images = images[val_size:]

        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(train_category_path, image))

        for image in val_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(val_category_path, image))


# 示例用法
source_directory = 'D:/ruikang/train'
train_directory = 'D:/ruikang/yaogan2/train'
val_directory = 'D:/ruikang/yaogan2/test'

split_dataset(source_directory, train_directory, val_directory, val_ratio=0.2)  #验证集比例为0.2