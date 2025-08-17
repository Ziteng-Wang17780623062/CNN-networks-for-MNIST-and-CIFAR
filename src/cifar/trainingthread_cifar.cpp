// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
#include "trainingthread_cifar.h"
#include <QThread>

TrainingThreadCIFAR10::TrainingThreadCIFAR10(CNN_CIFAR10* model, QObject* parent) : QThread(parent), m_model(model), m_device(torch::kCUDA) {
    if (!m_model) {
        emit errorOccurred("ģ��ָ��Ϊ�գ�");
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
        emit errorOccurred("ѵ�������������0��");
        return;
    }

    try {
        emit trainingStarted();

        if (m_device == torch::kCUDA && !torch::cuda::is_available()) {
            emit errorOccurred("����ʹ��CUDA���豸�����ã��Զ��л���CPU��");
            m_device = torch::kCPU;
            m_deviceStr = "CPU";
        }

        m_model->train(m_device, m_epochs, m_batches);

        emit trainingFinished();
    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch����: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("ѵ������: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("δ֪����");
    }
}

TrainingThreadCIFAR100::TrainingThreadCIFAR100(CNN_CIFAR100* model, QObject* parent) : QThread(parent), m_model(model), m_device(torch::kCUDA) {
    if (!m_model) {
        emit errorOccurred("ģ��ָ��Ϊ�գ�");
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
        emit errorOccurred("ѵ�������������0��");
        return;
    }

    try {
        emit trainingStarted();

        if (m_device == torch::kCUDA && !torch::cuda::is_available()) {
            emit errorOccurred("����ʹ��CUDA���豸�����ã��Զ��л���CPU��");
            m_device = torch::kCPU;
            m_deviceStr = "CPU";
        }

        m_model->train(m_device, m_epochs, m_batches);

        emit trainingFinished();
    } catch (const c10::Error& e) {
        emit errorOccurred(QString("LibTorch����: %1").arg(e.what()));
    } catch (const exception& e) {
        emit errorOccurred(QString("ѵ������: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("δ֪����");
    }
}
