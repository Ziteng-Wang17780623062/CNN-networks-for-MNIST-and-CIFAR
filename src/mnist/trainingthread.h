// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#ifndef TRAININGTHREAD_H
#define TRAININGTHREAD_H

#include <QThread>
#include <QString>
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include "model_mnist.h"

class TrainingThread : public QThread {
    Q_OBJECT

public:
    explicit TrainingThread(CNN_MNIST* model, QObject* parent);
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
    CNN_MNIST* m_model;
    QString m_modelName;
    QString m_deviceStr;
    int m_epochs;
    int m_batches;
    torch::Device m_device;
};

#endif // TRAININGTHREAD_H
