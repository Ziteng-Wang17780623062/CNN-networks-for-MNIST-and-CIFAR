// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
// testthread_cifar.h
#ifndef TESTTHREAD_CIFAR_H
#define TESTTHREAD_CIFAR_H

#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <QThread>
#include "model_cifar.h"

class TestThreadCIFAR10 : public QThread {
    Q_OBJECT

public:
    explicit TestThreadCIFAR10(CNN_CIFAR10 *model, torch::Device device, int batches, QObject *parent = nullptr);
    void setParameters(CNN_CIFAR10 *model, QString device, int batches);

signals:
    void testStarted();
    void testFinished(double acc);
    void errorOccurred(const QString &msg);

protected:
    void run() override;

private:
    CNN_CIFAR10 *m_model;
    torch::Device m_device;
    int m_batches;
};

class TestThreadCIFAR100 : public QThread {
    Q_OBJECT

public:
    explicit TestThreadCIFAR100(CNN_CIFAR100 *model, torch::Device device, int batches, QObject *parent = nullptr);
    void setParameters(CNN_CIFAR100 *model, QString device, int batches);

signals:
    void testStarted();
    void testFinished(double acc);
    void errorOccurred(const QString &msg);

protected:
    void run() override;

private:
    CNN_CIFAR100 *m_model;
    torch::Device m_device;
    int m_batches;
};

#endif // TESTTHREAD_CIFAR_H
