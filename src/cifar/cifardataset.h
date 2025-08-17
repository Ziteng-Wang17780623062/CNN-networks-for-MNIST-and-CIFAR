// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#ifndef CIFARDATASET_H
#define CIFARDATASET_H

#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

namespace cv { class Mat; }

class CIFARDataset : public torch::data::Dataset<CIFARDataset> {
public:
    enum class Mode { kTrain, kTest };

    struct Options {
        torch::Device device = torch::kCPU;
        bool normalize = true;
        bool shuffle = false;
    };

    CIFARDataset(const string& root, Mode mode, const Options& options, int num_classes);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override { return m_images.size(); }

    static torch::Tensor fromCVImage(const cv::Mat& image, const Options& options);

protected:
    vector<torch::Tensor> m_images;
    vector<int64_t> m_labels;
    torch::Device m_device;
    bool m_normalize;
    float m_mean[3] = {0.4914f, 0.4822f, 0.4465f};
    float m_std[3] = {0.2023f, 0.1994f, 0.2010f};

private:
    virtual void parseBinaryFiles(const string& root, Mode mode) { cout << "parseBinaryFiles in Base Class." << endl; };
};

class CIFAR10Dataset : public CIFARDataset {
public:
    CIFAR10Dataset(const string& root, Mode mode = Mode::kTrain, const Options& options = Options());

private:
    void parseBinaryFiles(const string& root, Mode mode) override;
};

class CIFAR100Dataset : public CIFARDataset {
public:
    CIFAR100Dataset(const string& root, Mode mode = Mode::kTrain, const Options& options = Options());

private:
    void parseBinaryFiles(const string& root, Mode mode) override;
};

#endif // CIFARDATASET_H
