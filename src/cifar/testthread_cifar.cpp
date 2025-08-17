// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#include "testthread_cifar.h"

TestThreadCIFAR10::TestThreadCIFAR10(CNN_CIFAR10 *model, torch::Device device, int batches, QObject *parent)
    : QThread(parent), m_model(model), m_device(device), m_batches(batches) {
    m_device = (device == torch::kCUDA && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TestThreadCIFAR10::setParameters(CNN_CIFAR10 *model, QString device, int batches) {
    m_model = model;
    m_batches = batches;
    m_device = (device == "NVIDIA GEFORCE RTX 4090 LapTop" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TestThreadCIFAR10::run() {
    emit testStarted();

    try {
        if (m_device == torch::kCUDA && !torch::cuda::is_available()) {
            emit errorOccurred("请求使用CUDA但设备不可用，自动切换至CPU！");
            m_device = torch::kCPU;
        }

        double acc = m_model->test(m_device, m_batches);

        emit testFinished(acc);

    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch错误: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("测试错误: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("未知错误");
    }
}

TestThreadCIFAR100::TestThreadCIFAR100(CNN_CIFAR100 *model, torch::Device device, int batches, QObject *parent)
    : QThread(parent), m_model(model), m_device(device), m_batches(batches) {
    m_device = (device == torch::kCUDA && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TestThreadCIFAR100::setParameters(CNN_CIFAR100 *model, QString device, int batches) {
    m_model = model;
    m_batches = batches;
    m_device = (device == "NVIDIA GEFORCE RTX 4090 LapTop" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TestThreadCIFAR100::run() {
    emit testStarted();

    try {
        if (m_device == torch::kCUDA && !torch::cuda::is_available()) {
            emit errorOccurred("请求使用CUDA但设备不可用，自动切换至CPU！");
            m_device = torch::kCPU;
        }

        double acc = m_model->test(m_device, m_batches);

        emit testFinished(acc);

    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch错误: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("测试错误: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("未知错误");
    }
}
