# CNN-networks-for-MNIST-and-CIFAR

（网络构建以及项目运行界面请参考\docs中的说明文件）

## 项目简介
本项目实现了基于卷积神经网络（CNN）的MNIST和CIFAR数据集图像分类模型，包含CNN_MNIST、CNN_CIFAR10和CNN_CIFAR100三种模型，支持模型训练、测试、保存和加载等功能，并提供了可视化的操作界面。

## 环境要求
- C++ 17及以上版本
- Qt 5/6（用于UI界面）
- PyTorch C++ API
- OpenCV（用于图像处理）
- 支持CUDA的GPU（可选，用于加速训练）

## 数据集和预训练模型获取
数据集（MNIST、CIFAR-10、CIFAR-100）和预训练模型已存储在百度网盘中，可通过以下链接下载：
- 链接: [https://pan.baidu.com/s/1Q2avRmZFAoBTQljV1kNlxg](https://pan.baidu.com/s/1Q2avRmZFAoBTQljV1kNlxg)
- 提取码: 0bug

## 数据集放置说明
1. 下载数据集压缩包并解压
2. 将MNIST数据集文件夹放置在项目根目录下的`../../data/MNIST/`路径中
3. 将CIFAR-10数据集文件夹（cifar-10-batches-bin）放置在项目根目录下的`../../data/`路径中
4. 将CIFAR-100数据集文件夹放置在项目根目录下的`../../data/`路径中

## 项目结构
- `main.cpp`：程序启动入口，负责初始化应用程序环境并创建主窗口实例
- `mainwindow.h` / `mainwindow.cpp`：主界面类的声明与实现，包含所有UI组件布局、用户交互响应及核心功能调度
- `model_mnist.h` / `model_mnist.cpp`：MNIST专用CNN模型实现，定义针对手写数字识别的网络结构，提供训练、推理、参数保存与加载接口
- `model_cifar.h` / `model_cifar.cpp`：CIFAR数据集模型实现，包含适用于CIFAR-10（10类）和CIFAR-100（100类）的卷积神经网络结构及相关操作方法
- `trainingthread.h` / `trainingthread.cpp`：MNIST模型训练线程，实现后台异步训练逻辑，避免UI界面卡顿，实时反馈训练进度
- `trainingthread_cifar.h` / `trainingthread_cifar.cpp`：CIFAR模型训练线程，专门处理CIFAR系列模型的异步训练任务，支持进度回调
- `testthread.h` / `testthread.cpp`：MNIST模型测试线程，在后台执行模型性能评估，计算测试集准确率并返回结果
- `testthread_cifar.h` / `testthread_cifar.cpp`：CIFAR模型测试线程，负责CIFAR模型的离线性能评估，生成测试报告
- `cifardataset.h` / `cifardataset.cpp`：CIFAR数据集处理工具，实现二进制数据解析、图像预处理、数据增强及批量加载功能
- `handwritingdialog.h` / `handwritingdialog.cpp`：手写输入交互模块，提供画板界面供用户手写数字，支持调用MNIST模型进行实时识别

## 模型说明
1. **CNN_MNIST**：针对MNIST手写数字数据集的CNN模型，包含2个卷积层和2个全连接层
2. **CNN_CIFAR10**：针对CIFAR-10数据集的CNN模型，包含多个卷积层、批归一化层、残差连接和注意力机制
3. **CNN_CIFAR100**：针对CIFAR-100数据集的CNN模型，结构与CNN_CIFAR10类似，适配100个类别的分类任务

## 使用方法
1. 编译项目生成可执行文件
2. 运行程序，在主界面中进行以下操作：
   - 选择模型（CNN_MNIST、CNN_CIFAR10、CNN_CIFAR100）
   - 选择计算设备（CPU或GPU）
   - 设置训练轮数（Epochs）和批处理量（Batches）
   - 点击"训练模型"开始训练
   - 训练完成后可点击"测试模型"评估模型性能
   - 可通过"保存模型参数"和"加载模型参数"按钮进行模型的持久化操作
   - 点击"使用模型"可对新的图像进行分类预测

## 功能说明
- **模型训练**：支持在指定设备上以设定的参数训练模型，实时显示训练进度、损失值和准确率
- **模型测试**：使用测试集评估训练好的模型性能，输出准确率
- **模型预测**：对输入的图像进行分类预测
- **模型保存/加载**：支持将训练好的模型参数保存到文件或从文件加载

## 注意事项
1. 首次运行时若数据集路径不存在，程序会尝试自动创建数据目录，请将数据集放入相应目录
2. 使用GPU训练需要确保CUDA环境配置正确且PyTorch支持CUDA
3. 训练过程中日志会实时显示在界面的日志区域，可通过"清空日志"按钮清除日志内容
