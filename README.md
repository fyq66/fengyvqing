# fengyvqing
遥感图像的分类（2024睿抗机器人开发者大赛全国总决赛国一项目）


赛事提供的遥感图像数据集总共包含5个类：airplane,bridge,palace,ship,stadium,自取如下（以划分训练集与测试集）
链接：https://pan.baidu.com/s/1AyzFxW3r92Xk16aNTrKUTA 
提取码：nfeu 
--来自百度网盘超级会员V2的分享


split.py：用于划分数据集的训练集与测试集，测试集比例为0.2
train.py:用于训练，accuracy可以达到99+,val_acc可以达到1.0，竞赛综合成绩为97.788
predict.py:用于对模型性能的评估与预测
