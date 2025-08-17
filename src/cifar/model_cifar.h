// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#ifndef MODEL_CIFAR_H
#define MODEL_CIFAR_H

#undef slots
#include <torch/torch.h>
#include <torch/serialize/archive.h>
#define slots Q_SLOTS

#include <QObject>

using namespace std;

class CNN_CIFAR10 : public QObject, public torch::nn::Module {
    Q_OBJECT
public:
    CNN_CIFAR10(int num_classes, int input_channels);
    torch::Tensor forward(torch::Tensor x);
    void train(torch::Device device, int epochs, int batch_size);
    double test(torch::Device device, int batch_size);
    int predict(torch::Tensor input);
    torch::Device device() const;

    static string classify(int result);

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;
    void save(const string& path);
    void load(const string& path);

signals:
    void epochCompleted(int total, int epoch, double loss, double accuracy);

private:
    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::BatchNorm2d bn1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::BatchNorm2d bn1_2{nullptr};

    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::BatchNorm2d bn2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::BatchNorm2d bn2_2{nullptr};
    torch::nn::Conv2d conv2_3{nullptr};
    torch::nn::BatchNorm2d bn2_3{nullptr};

    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::BatchNorm2d bn3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::BatchNorm2d bn3_2{nullptr};
    torch::nn::Conv2d conv3_3{nullptr};
    torch::nn::BatchNorm2d bn3_3{nullptr};

    torch::nn::Conv2d conv4_1{nullptr};
    torch::nn::BatchNorm2d bn4_1{nullptr};
    torch::nn::Conv2d conv4_2{nullptr};
    torch::nn::BatchNorm2d bn4_2{nullptr};

    torch::nn::Conv2d residual1{nullptr};
    torch::nn::BatchNorm2d residual_bn1{nullptr};
    torch::nn::Conv2d residual2{nullptr};
    torch::nn::BatchNorm2d residual_bn2{nullptr};

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    torch::nn::Dropout2d spatial_dropout{nullptr};
    torch::nn::Dropout dropout{nullptr};

    torch::nn::AdaptiveAvgPool2d global_pool{nullptr};
    torch::nn::Linear attention_fc1{nullptr};
    torch::nn::Linear attention_fc2{nullptr};

    torch::Tensor channel_attention(torch::Tensor x);

    torch::Device m_device;
};

class CNN_CIFAR100 : public QObject, public torch::nn::Module {
    Q_OBJECT
public:
    CNN_CIFAR100(int num_classes, int input_channels);
    torch::Tensor forward(torch::Tensor x);
    void train(torch::Device device, int epochs, int batch_size);
    double test(torch::Device device, int batch_size);
    int predict(torch::Tensor input);
    torch::Device device() const;

    static string classify(int result);

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;
    void save(const string& path);
    void load(const string& path);

signals:
    void epochCompleted(int total, int epoch, double loss, double accuracy);
    void errorOccurred(const QString& errorMessage);

private:
    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::BatchNorm2d bn1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::BatchNorm2d bn1_2{nullptr};

    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::BatchNorm2d bn2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::BatchNorm2d bn2_2{nullptr};
    torch::nn::Conv2d conv2_3{nullptr};
    torch::nn::BatchNorm2d bn2_3{nullptr};

    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::BatchNorm2d bn3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::BatchNorm2d bn3_2{nullptr};
    torch::nn::Conv2d conv3_3{nullptr};
    torch::nn::BatchNorm2d bn3_3{nullptr};

    torch::nn::Conv2d conv4_1{nullptr};
    torch::nn::BatchNorm2d bn4_1{nullptr};
    torch::nn::Conv2d conv4_2{nullptr};
    torch::nn::BatchNorm2d bn4_2{nullptr};

    torch::nn::Conv2d residual1{nullptr};
    torch::nn::BatchNorm2d residual_bn1{nullptr};
    torch::nn::Conv2d residual2{nullptr};
    torch::nn::BatchNorm2d residual_bn2{nullptr};

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    torch::nn::Dropout2d spatial_dropout{nullptr};
    torch::nn::Dropout dropout{nullptr};

    torch::nn::AdaptiveAvgPool2d global_pool{nullptr};
    torch::nn::Linear attention_fc1{nullptr};
    torch::nn::Linear attention_fc2{nullptr};

    torch::Tensor channel_attention(torch::Tensor x);

    torch::Device m_device;
};

#endif // MODEL_CIFAR_H
