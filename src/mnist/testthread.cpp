// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
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
            emit errorOccurred("����ʹ��CUDA���豸�����ã��Զ��л���CPU��");
            m_device = torch::kCPU;
        }

        double acc = m_model->test(m_device, m_batches);

        emit testFinished(acc);

    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch����: %1").arg(e.what()));
    } catch (const std::exception& e) {
        emit errorOccurred(QString("Training����: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("Unkown����");
    }
}
