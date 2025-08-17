// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#include "cifardataset.h"
#include <filesystem>
#include <fstream>

CIFARDataset::CIFARDataset(const string& root, Mode mode, const Options& options, int num_classes)
    : m_device(options.device), m_normalize(options.normalize) {
    if (!filesystem::exists(root)) {
        throw runtime_error("Dataset path not found: " + root);
    }
    if (m_images.size() != m_labels.size()) {
        throw runtime_error("Image and label count mismatch");
    }
}

torch::data::Example<> CIFARDataset::get(size_t index) {
    auto image = m_images.at(index).to(m_device);
    auto label = torch::tensor(m_labels.at(index), torch::kInt64).to(m_device);
    return {image, label};
}

torch::Tensor CIFARDataset::fromCVImage(const cv::Mat& image, const Options& options) {
    if (image.channels() != 3) {
        cerr << "Error: Image must have 3 channels (BGR or RGB)" << endl;
    }

    vector<cv::Mat> channels;
    split(image, channels);

    namedWindow("Channels", cv::WINDOW_NORMAL);
    cv::resizeWindow("Channels", 1200, 400);

    cv::Mat display;
    hconcat(channels, display);

    putText(display, "Blue Channel", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    putText(display, "Green Channel", cv::Point(image.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    putText(display, "Red Channel", cv::Point(2 * image.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    imshow("Channels", display);
    cv::waitKey(0);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(32, 32));

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    return torch::from_blob(resized.data, {1, 32, 32, 3}, torch::kFloat32).permute({0, 3, 1, 2}).to(options.device);
}

CIFAR10Dataset::CIFAR10Dataset(const string& root, Mode mode, const Options& options)
    : CIFARDataset(root, mode, options, 10) {
    CIFAR10Dataset::parseBinaryFiles(root, mode);
    m_mean[0] = 0.5071f; m_mean[1] = 0.4867f; m_mean[2] = 0.4408f;
    m_std[0] = 0.2675f;  m_std[1] = 0.2565f;  m_std[2] = 0.2761f;
}

void CIFAR10Dataset::parseBinaryFiles(const string& root, Mode mode) {
    cout << "parseBinaryFiles in CIFAR10Dataset Class." << endl;

    const int IMAGE_SIZE = 32 * 32 * 3;
    vector<string> filenames;

    if (mode == Mode::kTrain) {
        for (int i = 1; i <= 5; ++i) {
            filenames.push_back(root + "/data_batch_" + to_string(i) + ".bin");
        }
    } else {
        filenames.push_back(root + "/test_batch" + ".bin");
    }

    for (const auto& file : filenames) {
        ifstream ifs(file, ios::binary);
        if (!filesystem::exists(root)) {
            cout << "Dataset path not found: " << root << endl;
            throw runtime_error("Dataset path not found: " + root);
        }
        if (!ifs.is_open()) {
            cout << "Failed to open file: " << file << endl;
            throw runtime_error("Failed to open file: " + file);
        }
        char label;
        vector<char> image_data(IMAGE_SIZE);

        while (ifs.read(&label, 1) && ifs.read(image_data.data(), IMAGE_SIZE)) {
            torch::Tensor image = torch::from_blob(image_data.data(), {3, 32, 32}, torch::kUInt8).to(torch::kFloat32).div(255.0);

            if (m_normalize) {
                for (int c = 0; c < 3; ++c) {
                    image[c] = (image[c] - m_mean[c]) / m_std[c];
                }
            }

            m_images.push_back(image);
            m_labels.push_back(static_cast<int64_t>(static_cast<unsigned char>(label)));
        }

        if (!ifs.eof()) {
            throw runtime_error("File read error: " + file);
        }
    }

    cout << "parseBinaryFiles in CIFAR10Dataset completed." << endl;
}

CIFAR100Dataset::CIFAR100Dataset(const string& root, Mode mode, const Options& options)
    : CIFARDataset(root, mode, options, 100) {
    CIFAR100Dataset::parseBinaryFiles(root, mode);
    m_mean[0] = 0.5071f; m_mean[1] = 0.4867f; m_mean[2] = 0.4408f;
    m_std[0] = 0.2675f;  m_std[1] = 0.2565f;  m_std[2] = 0.2761f;
}

void CIFAR100Dataset::parseBinaryFiles(const string& root, Mode mode) {
    cout << "parseBinaryFiles in CIFAR100Dataset Class." << endl;

    const int IMAGE_SIZE = 32 * 32 * 3;
    string filename = (mode == Mode::kTrain) ? "train" : "test";
    filename = root + "/" + filename + ".bin";

    ifstream ifs(filename, ios::binary);
    if (!ifs.is_open()) {
        throw runtime_error("Failed to open file: " + filename + ".bin");
    }

    struct CIFAR100Record {
        char coarse_label;
        char fine_label;
        char image_data[IMAGE_SIZE];
    };

    CIFAR100Record record;
    while (ifs.read(reinterpret_cast<char*>(&record), sizeof(CIFAR100Record))) {
        torch::Tensor image = torch::from_blob(record.image_data, {3, 32, 32}, torch::kUInt8)
        .to(torch::kFloat32)
            .div(255.0);

        if (m_normalize) {
            for (int c = 0; c < 3; ++c) {
                image[c] = (image[c] - m_mean[c]) / m_std[c];
            }
        }

        m_images.push_back(image);
        m_labels.push_back(static_cast<int64_t>(static_cast<unsigned char>(record.fine_label)));
    }
    cout<<"TEST"<<endl;
    if (!ifs.eof()) {
        throw runtime_error("File read error: " + filename + ".bin");
    }
}
