// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
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
            emit errorOccurred("����ʹ��CUDA���豸�����ã��Զ��л���CPU��");
            m_device = torch::kCPU;
        }

        double acc = m_model->test(m_device, m_batches);

        emit testFinished(acc);

    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch����: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("���Դ���: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("δ֪����");
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
            emit errorOccurred("����ʹ��CUDA���豸�����ã��Զ��л���CPU��");
            m_device = torch::kCPU;
        }

        double acc = m_model->test(m_device, m_batches);

        emit testFinished(acc);

    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch����: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("���Դ���: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("δ֪����");
    }
}
