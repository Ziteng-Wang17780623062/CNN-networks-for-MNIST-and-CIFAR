// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
#ifndef TRAININGTHREAD_CIFAR_H
#define TRAININGTHREAD_CIFAR_H

#include <QThread>
#include <QString>
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include "model_cifar.h"

class TrainingThreadCIFAR10 : public QThread {
    Q_OBJECT

public:
    explicit TrainingThreadCIFAR10(CNN_CIFAR10* model, QObject* parent);
    void setParameters(const QString& modelName, const QString& device, int epochs, int batches);

signals:
    void trainingStarted();
    void progressUpdated(int progress);
    void epochCompleted(int total, int epoch, double loss, double accuracy);
    void trainingFinished();
    void errorOccurred(const QString& message);

protected:
    void run() override;

private:
    CNN_CIFAR10* m_model;
    QString m_modelName;
    QString m_deviceStr;
    int m_epochs;
    int m_batches;
    torch::Device m_device;
};

class TrainingThreadCIFAR100 : public QThread {
    Q_OBJECT

public:
    explicit TrainingThreadCIFAR100(CNN_CIFAR100* model, QObject* parent);
    void setParameters(const QString& modelName, const QString& device, int epochs, int batches);

signals:
    void trainingStarted();
    void progressUpdated(int progress);
    void epochCompleted(int total, int epoch, double loss, double accuracy);
    void trainingFinished();
    void errorOccurred(const QString& message);

protected:
    void run() override;

private:
    CNN_CIFAR100* m_model;
    QString m_modelName;
    QString m_deviceStr;
    int m_epochs;
    int m_batches;
    torch::Device m_device;
};

#endif // TRAININGTHREAD_CIFAR_H
