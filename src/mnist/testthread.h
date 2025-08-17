// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
#ifndef TESTTHREAD_H
#define TESTTHREAD_H

#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <QThread>
#include "model_mnist.h"

class TestThread : public QThread {
    Q_OBJECT

public:
    explicit TestThread(CNN_MNIST *model, torch::Device device, int batches, QObject *parent = nullptr);
    void setModel(CNN_MNIST *model);
    void setParameters(CNN_MNIST *model, QString device, int batches);

signals:
    void testStarted();
    void testFinished(double acc);
    void errorOccurred(const QString &msg);

protected:
    void run() override;

private:
    CNN_MNIST *m_model;
    torch::Device m_device;
    int m_batches;
};

#endif // TESTTHREAD_H
