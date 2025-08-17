// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#include "trainingthread_cifar.h"
#include <QThread>

TrainingThreadCIFAR10::TrainingThreadCIFAR10(CNN_CIFAR10* model, QObject* parent) : QThread(parent), m_model(model), m_device(torch::kCUDA) {
    if (!m_model) {
        emit errorOccurred("模型指针为空！");
        return;
    }

    connect(m_model, &CNN_CIFAR10::epochCompleted, [this](int total, int epoch, double loss, double accuracy) {
        emit epochCompleted(total, epoch, loss, accuracy);
    });
}

void TrainingThreadCIFAR10::setParameters(const QString& modelName, const QString& device, int epochs, int batches) {
    m_modelName = modelName;
    m_deviceStr = device;
    m_epochs = epochs;
    m_batches = batches;
    m_device = (device == "NVIDIA GEFORCE RTX 4090 LapTop" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TrainingThreadCIFAR10::run() {
    if (m_epochs <= 0) {
        emit errorOccurred("训练轮数必须大于0！");
        return;
    }

    try {
        emit trainingStarted();

        if (m_device == torch::kCUDA && !torch::cuda::is_available()) {
            emit errorOccurred("请求使用CUDA但设备不可用，自动切换至CPU！");
            m_device = torch::kCPU;
            m_deviceStr = "CPU";
        }

        m_model->train(m_device, m_epochs, m_batches);

        emit trainingFinished();
    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch错误: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("训练错误: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("未知错误");
    }
}

TrainingThreadCIFAR100::TrainingThreadCIFAR100(CNN_CIFAR100* model, QObject* parent) : QThread(parent), m_model(model), m_device(torch::kCUDA) {
    if (!m_model) {
        emit errorOccurred("模型指针为空！");
        return;
    }

    connect(m_model, &CNN_CIFAR100::epochCompleted, [this](int total, int epoch, double loss, double accuracy) {
        emit epochCompleted(total, epoch, loss, accuracy);
    });
}

void TrainingThreadCIFAR100::setParameters(const QString& modelName, const QString& device, int epochs, int batches) {
    m_modelName = modelName;
    m_deviceStr = device;
    m_epochs = epochs;
    m_batches = batches;
    m_device = (device == "NVIDIA GEFORCE RTX 4090 LapTop" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TrainingThreadCIFAR100::run() {
    if (m_epochs <= 0) {
        emit errorOccurred("训练轮数必须大于0！");
        return;
    }

    try {
        emit trainingStarted();

        if (m_device == torch::kCUDA && !torch::cuda::is_available()) {
            emit errorOccurred("请求使用CUDA但设备不可用，自动切换至CPU！");
            m_device = torch::kCPU;
            m_deviceStr = "CPU";
        }

        m_model->train(m_device, m_epochs, m_batches);

        emit trainingFinished();
    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch错误: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("训练错误: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("未知错误");
    }
}
