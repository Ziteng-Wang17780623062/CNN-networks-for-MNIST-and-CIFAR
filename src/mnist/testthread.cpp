// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#include "testthread.h"

TestThread::TestThread(CNN_MNIST *model, torch::Device device, int batches, QObject *parent)
    : QThread(parent), m_model(model), m_device(device), m_batches(batches)
{
    m_device = (device == torch::kCUDA && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TestThread::setModel(CNN_MNIST *model) {
    m_model = model;
}

void TestThread::setParameters(CNN_MNIST *model, QString device, int batches) {
    setModel(model);
    m_batches = batches;
    m_device = (device == "NVIDIA GEFORCE RTX 4090 LapTop" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
}

void TestThread::run() {
    emit testStarted();

    try {
        emit testStarted();

        if (m_device == torch::kCUDA && !torch::cuda::is_available()) {
            emit errorOccurred("请求使用CUDA但设备不可用，自动切换至CPU！");
            m_device = torch::kCPU;
        }

        double acc = m_model->test(m_device, m_batches);

        emit testFinished(acc);

    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch错误: %1").arg(e.what()));
    } catch (const std::exception& e) {
        emit errorOccurred(QString("Training错误: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("Unkown错误");
    }
}
