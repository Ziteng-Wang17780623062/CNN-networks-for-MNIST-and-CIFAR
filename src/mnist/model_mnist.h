// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
#ifndef MODEL_MNIST_H
#define MODEL_MNIST_H

#undef slots
#include <torch/torch.h>
#include <torch/serialize/archive.h>
#define slots Q_SLOTS

#include <QObject>

using namespace std;

class CNN_MNIST : public QObject, public torch::nn::Module {
    Q_OBJECT
public:
    CNN_MNIST(int num_classes, int input_channels);
    torch::Tensor forward(torch::Tensor x);
    void train(torch::Device device, int epochs, int batch_size);
    double test(torch::Device device, int batch_size);
    int predict(torch::Tensor input);
    torch::Device device() const;

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;
    void save(const string& path);
    void load(const string& path);

signals:
    void epochCompleted(int taotal, int epoch, double loss, double accuracy);

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    torch::Device m_device;
};

#endif // MODEL_MNIST_H
