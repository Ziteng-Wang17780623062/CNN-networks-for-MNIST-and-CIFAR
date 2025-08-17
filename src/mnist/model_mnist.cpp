// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
#include "model_mnist.h"
#include <filesystem>

CNN_MNIST::CNN_MNIST(int num_classes, int input_channels) : m_device(torch::kCUDA) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 32, 3).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
    fc1 = register_module("fc1", torch::nn::Linear(64 * 7 * 7, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, num_classes));

    if (torch::cuda::is_available()) {
        m_device = torch::kCUDA;
        cout << "CUDA is available! Using GPU auto." << endl;
    }
    else {
        m_device = torch::kCPU;
        cout << "CUDA not available. Using CPU auto." << endl;
    }
}

torch::Tensor CNN_MNIST::forward(torch::Tensor x) {
    x = torch::relu(conv1->forward(x));
    x = torch::max_pool2d(x, 2);

    x = torch::relu(conv2->forward(x));
    x = torch::max_pool2d(x, 2);

    x = x.view({-1, 64 * 7 * 7});

    x = torch::relu(fc1->forward(x));

    x = fc2->forward(x);

    return x;
}

void CNN_MNIST::train(torch::Device device, int epochs, int batch_size) {
    cout << "Current path: " << filesystem::current_path() << endl;
    if (!filesystem::exists("../../data")) {
        cerr << "�������ݼ�·�������ڣ����Դ���Ŀ¼..." << endl;
        try {
            filesystem::create_directory("../../data");
            cout << "�Ѵ�������Ŀ¼���뽫MNIST���ݼ��������С�" << endl;
        } catch (const exception& e) {
            cerr << "����Ŀ¼ʧ�ܣ�" << e.what() << endl;
            return;
        }
    }

    if (device == torch::kCUDA && torch::cuda::is_available()) {
        device = torch::kCUDA;
        cout << "CUDA is available! Using GPU." << endl;
    }
    else {
        device = torch::kCPU;
        cout << "CUDA not available. Using CPU." << endl;
    }
    m_device = device;

    this->to(m_device);

    try {
        auto train_dataset = torch::data::datasets::MNIST("../../data/MNIST/", torch::data::datasets::MNIST::Mode::kTrain).map(torch::data::transforms::Normalize<>(0.1307, 0.3081)).map(torch::data::transforms::Stack<>());
        const size_t train_dataset_size = train_dataset.size().value();
        cout << "train_dataset_size: " << train_dataset_size << endl;
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

        torch::optim::Adam optimizer(this->parameters(), 0.001);

        cout << "Start training..." << endl;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float running_loss = 0.0;
            int64_t correct = 0;
            int64_t total = 0;
            int64_t batch_count = 0;

            this->torch::nn::Module::train();

            for (auto const &batch : *train_loader) {
                auto data = batch.data.to(device);
                auto targets = batch.target.to(device);

                optimizer.zero_grad();

                auto output = this->forward(data);

                auto loss = torch::cross_entropy_loss(output, targets);

                loss.backward();

                optimizer.step();

                running_loss += loss.item<double>();
                auto predicted = output.argmax(1);
                total += targets.size(0);
                correct += predicted.eq(targets).sum().item<int64_t>();

                batch_count++;
            }

            cout << "Epoch [" << epoch << "/" << epochs << "], " << "Loss: " << fixed << setprecision(4) << running_loss / batch_count << ", " << "Accuracy: " << static_cast<double>(correct) / total << endl;

            emit epochCompleted(epochs, epoch, running_loss / batch_count, static_cast<double>(correct) / total);
        }
        cout << "Finish training!" << endl;
    } catch (const c10::Error& e) {
        cerr << "C10 Error: " << e.what() << endl;
    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
    }
}

double CNN_MNIST::test(torch::Device device, int batch_size) {
    if (device == torch::kCUDA && torch::cuda::is_available()) {
        device = torch::kCUDA;
        cout << "CUDA is available! Using GPU." << endl;
    }
    else {
        device = torch::kCPU;
        cout << "CUDA not available. Using CPU." << endl;
    }
    m_device = device;

    this->to(m_device);

    auto test_dataset = torch::data::datasets::MNIST("../../data/MNIST/", torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Normalize<>(0.1307, 0.3081)).map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), batch_size);

    cout << "�������ݼ������" << endl;

    this->eval();

    int64_t correct = 0;
    int64_t total = 0;

    torch::NoGradGuard no_grad;

    for (const auto &batch : *test_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        auto output = this->forward(data);
        auto predicted = output.argmax(1);

        total += targets.size(0);
        correct += predicted.eq(targets).sum().item<int64_t>();
    }

    double accuracy = static_cast<double>(correct) / total;
    cout << "Accuracy: " << accuracy << endl;

    return accuracy;
}

int CNN_MNIST::predict(torch::Tensor input) {
    this->eval();
    torch::Tensor output = forward(input);
    if (m_device == torch::kCUDA) {
        output = output.to(torch::kCPU);
    }
    cout << output.transpose(-1, 1) << endl;
    return output.argmax(1).item<int>();
}

torch::Device CNN_MNIST::device() const {
    return m_device;
}

void CNN_MNIST::save(torch::serialize::OutputArchive& archive) const {
    torch::nn::Module::save(archive);
    archive.write("device", m_device.type() == torch::kCUDA ? "cuda" : "cpu");
}

void CNN_MNIST::load(torch::serialize::InputArchive& archive) {
    torch::nn::Module::load(archive);
    c10::IValue ivalue_device;
    if (archive.try_read("device", ivalue_device)) {
        if (ivalue_device.isString()) {
            string device_str = ivalue_device.toStringRef();
            m_device = (device_str == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
            this->to(m_device);
        }
    }
}

void CNN_MNIST::save(const string& path) {
    try {
        torch::serialize::OutputArchive archive;
        torch::nn::Module::save(archive);
        archive.write("device", m_device.type() == torch::kCUDA ? "cuda" : "cpu");
        archive.save_to(path);
        cout << "ģ�ͱ���ɹ�: " << path << endl;
    } catch (const exception& e) {
        cerr << "����ʧ��: " << e.what() << endl;
        throw;
    }
}

void CNN_MNIST::load(const string& path) {
    try {
        if (!filesystem::exists(path)) {
            throw runtime_error("�ļ�������: " + path);
        }

        torch::serialize::InputArchive archive;
        archive.load_from(path);
        torch::nn::Module::load(archive);

        c10::IValue device_str;
        if (archive.try_read("device", device_str)) {
            m_device = (device_str == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
            this->to(m_device);
        }
        cout << "ģ�ͼ��سɹ�: " << path << endl;
    } catch (const exception& e) {
        cerr << "����ʧ��: " << e.what() << endl;
        throw;
    }
}
