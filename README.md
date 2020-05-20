# 基于 ResNet 的交通标识分类 Chinese-Traffic-Signs-Classify based on ResNet
## 网络 NN
模型使用 ResNet50

Based on ResNet50

## 数据集 dataset
数据来源于 [Chinese Traffic Sign Database](http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html)

共 4170 张训练图片，1994 张测试图片，58 个类别

数据集很小，而且分类数量不均衡，最少的一类只有 2 张训练图片

将 Train 和 Test 文件夹下的 `label.txt` 文件改为 `path label` 的格式

All the images comes from [Chinese Traffic Sign Database](http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html)

There are 4,170 images for training and 1,994 images for testing in the database, containing 58 categories.

The amount of images is quite small and unevenly classified.

Format `label.txt` under Train and Test folder to `path label`.

## 模型数据

|test accurary|precision|recall|f1|
|--|--|--|--|
|92%|0.9784|0.9902|0.9781|

AdamW + 0.01_lr with 0.5_gamma/5_epoch：
<img src="model&img/score-adamw-0.01lr-50epoch.svg" alt="Adamw" style="zoom:50%;" />

SGD + 0.01_lr with 0.5_gamma/10_epoch：
<img src="model&img/score-SGD-0.01lr-50epoch.svg" alt="SGD" style="zoom:50%;" />

forecast result：
<img src="model&img/result.png" alt="forecast result" style="zoom:67%;" />

