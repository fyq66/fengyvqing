import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

# 加载模型的路径,加载请注意 path 是相对路径, 与当前文件同级。
path = "results/model_cnn.h5"
# 加载模型，不编译
model = tf.keras.models.load_model(path, compile=False)

# 重新编译模型，使用标准的SGD优化器
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 设定测试集目录
test_dir = 'yaoganshujvji/test'

# 标签名称列表，必须与训练时保持一致
label = ['airplane', 'bridge', 'palace', 'ship', 'stadium']
im_size = 224


def predict(X):
    """
    使用模型对单张图像进行预测
    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据， shape(256, 256, 3)。
    return：
        y_predict : str, 预测的类别标签。
    """
    # 数据预处理
    X = cv2.resize(X, (im_size, im_size))  # 调整图像大小到模型输入要求的大小
    X = X.reshape(1, im_size, im_size, 3)  # 将图像转换为模型所需的4维输入格式
    X = X / 255.0  # 归一化像素值到0-1之间

    # 模型预测
    prediction = model.predict(X)
    y_predict = np.argmax(prediction)  # 获取预测结果的最大概率对应的类别索引
    y_predict = label[y_predict]  # 将类别索引转换为标签名称

    return y_predict


def evaluate_model_accuracy():
    """
    评估模型在测试集上的准确率
    """
    y_true = []  # 存储真实标签
    y_pred = []  # 存储预测标签

    # 遍历测试集的每个类别文件夹
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # 跳过非目录项

        # 遍历类别文件夹中的每个图像
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                continue  # 跳过无法读取的图像

            # 获取预测标签
            pred_label = predict(image)
            y_pred.append(pred_label)  # 记录预测标签
            y_true.append(class_name)  # 记录真实标签

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"模型在测试集上的准确率为: {accuracy * 100:.2f}%")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=label))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=label)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()


# 主函数调用
if __name__ == "__main__":
    evaluate_model_accuracy()




