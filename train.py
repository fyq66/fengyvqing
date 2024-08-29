# 导入所需的库
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

# 设置训练和测试数据集的目录路径
train_dir = 'yaoganshujvji/train'  # 训练数据集目录
test_dir = 'yaoganshujvji/test'    # 测试/验证数据集目录

# 数据增强：定义训练数据生成器
train_images = ImageDataGenerator(
    rescale=1/255,                # 将图像像素值缩放到0-1之间
    rotation_range=20,            # 随机旋转角度范围
    width_shift_range=0.1,        # 水平平移范围
    height_shift_range=0.1,       # 垂直平移范围
    shear_range=0.1,              # 剪切变换范围
    zoom_range=0.1,               # 缩放变换范围
    horizontal_flip=True,         # 随机水平翻转
    vertical_flip=True,           # 随机垂直翻转
    brightness_range=[0.9, 1.1],  # 亮度调节范围
    channel_shift_range=0.1,      # 通道值平移范围
    fill_mode='nearest'           # 填充模式
)

# 定义测试/验证数据生成器（仅做缩放）
test_images = ImageDataGenerator(rescale=1/255)

# 设置图像尺寸和批量大小
im_size = 224
batch_size = 32

# 从目录加载训练数据，并应用数据增强
train_gen = train_images.flow_from_directory(
    directory=train_dir,           # 训练数据目录
    batch_size=batch_size,         # 每批次加载的图像数量
    shuffle=True,                  # 是否打乱数据顺序
    target_size=(im_size, im_size),# 调整图像到指定大小
    class_mode='sparse'            # 使用稀疏标签，适用于多分类问题
)

# 从目录加载测试/验证数据
val_gen = test_images.flow_from_directory(
    directory=test_dir,            # 测试/验证数据目录
    batch_size=batch_size,         # 每批次加载的图像数量
    shuffle=False,                 # 不打乱数据顺序
    target_size=(im_size, im_size),# 调整图像到指定大小
    class_mode='sparse'            # 使用稀疏标签，适用于多分类问题
)

# 使用预训练的DenseNet201作为基础模型，不包含顶部分类层
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

# 构建自定义的模型
def build_model(learning_rate=0.001):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)              # 添加全局平均池化层
    # 以下为可选层，暂时注释掉
    # x = Dense(8192, activation='relu')(x)       # 全连接层，使用ReLU激活函数
    # x = BatchNormalization()(x)                 # 批归一化层
    # x = Dropout(0.4)(x)                         # Dropout层，防止过拟合
    # x = Dense(4096, activation='relu')(x)       # 全连接层，使用ReLU激活函数
    # x = BatchNormalization()(x)                 # 批归一化层
    # x = Dropout(0.4)(x)                         # Dropout层，防止过拟合
    # 定义输出层，使用softmax激活函数，适用于多分类
    predictions = tf.keras.layers.Dense(45, activation='softmax')(x)
    # 创建模型对象
    model = Model(inputs=base_model.input, outputs=predictions)

    # 定义优化器，使用带动量的随机梯度下降（SGD），启用Nesterov加速
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    # 编译模型，使用稀疏分类交叉熵作为损失函数，监控准确率
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model

# 定义学习率调度策略：Cosine Annealing with Restarts（余弦退火与重启）
def cosine_annealing_schedule_with_restarts(epoch, lr):
    initial_lr = 0.001  # 初始学习率
    T_max = 50          # 完整的学习率周期
    T_cur = epoch % T_max  # 当前周期中的步数
    # 计算学习率：余弦函数实现周期性变化
    return 0.5 * initial_lr * (1 + np.cos(np.pi * T_cur / T_max))

# 定义学习率调度回调
lr_scheduler = LearningRateScheduler(cosine_annealing_schedule_with_restarts)

# 定义早停回调：监控val_loss，当20个epoch不再改善时停止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 定义模型检查点回调：当val_acc提高时，保存模型权重
model_checkpoint = ModelCheckpoint(filepath='result1/model_cnn.h5', monitor='val_acc', save_best_only=True, verbose=1)

# 定义TensorBoard回调，用于可视化训练过程
tensorboard = TensorBoard(log_dir='./logs')

# 构建并编译模型
model = build_model()

# 训练模型
history = model.fit(
    train_gen,                   # 训练数据生成器
    epochs=150,                  # 训练轮数
    validation_data=val_gen,     # 验证数据生成器
    callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard] # 回调列表
)

# 保存最终训练好的模型
model.save('result1/final_model_cnn.h5')
