// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
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
