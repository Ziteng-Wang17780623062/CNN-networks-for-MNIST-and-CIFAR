// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#include "model_cifar.h"
#include "cifardataset.h"
#include <filesystem>

CNN_CIFAR10::CNN_CIFAR10(int num_classes, int input_channels) : m_device(torch::kCUDA) {
    try {
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 64, 3).padding(1)));
        bn1_1 = register_module("bn1_1", torch::nn::BatchNorm2d(64));
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        bn1_2 = register_module("bn1_2", torch::nn::BatchNorm2d(64));

        conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1).stride(2)));
        bn2_1 = register_module("bn2_1", torch::nn::BatchNorm2d(128));
        conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
        bn2_2 = register_module("bn2_2", torch::nn::BatchNorm2d(128));
        conv2_3 = register_module("conv2_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
        bn2_3 = register_module("bn2_3", torch::nn::BatchNorm2d(128));

        conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1).stride(2)));
        bn3_1 = register_module("bn3_1", torch::nn::BatchNorm2d(256));
        conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        bn3_2 = register_module("bn3_2", torch::nn::BatchNorm2d(256));
        conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        bn3_3 = register_module("bn3_3", torch::nn::BatchNorm2d(256));

        conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
        bn4_1 = register_module("bn4_1", torch::nn::BatchNorm2d(512));
        conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
        bn4_2 = register_module("bn4_2", torch::nn::BatchNorm2d(512));

        residual1 = register_module("residual1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1).stride(2)));
        residual_bn1 = register_module("residual_bn1", torch::nn::BatchNorm2d(128));
        residual2 = register_module("residual2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1).stride(2)));
        residual_bn2 = register_module("residual_bn2", torch::nn::BatchNorm2d(256));

        global_pool = register_module("global_pool", torch::nn::AdaptiveAvgPool2d(1));
        attention_fc1 = register_module("attention_fc1", torch::nn::Linear(512, 128));
        attention_fc2 = register_module("attention_fc2", torch::nn::Linear(128, 512));

        fc1 = register_module("fc1", torch::nn::Linear(512 * 4 * 4, 1024));
        fc2 = register_module("fc2", torch::nn::Linear(1024, 512));
        fc3 = register_module("fc3", torch::nn::Linear(512, num_classes));

        spatial_dropout = register_module("spatial_dropout", torch::nn::Dropout2d(0.1));
        dropout = register_module("dropout", torch::nn::Dropout(0.3));
    }
    catch (const exception& e) {
        cerr<<"ERROR CNN_CIFAR10"<<e.what()<<endl;
        throw;
    }

    if (torch::cuda::is_available()) {
        m_device = torch::kCUDA;
        cout << "CUDA is available! Using GPU auto." << endl;
    }
    else {
        m_device = torch::kCPU;
        cout << "CUDA not available. Using CPU auto." << endl;
    }

    this->to(m_device);
}

torch::Tensor CNN_CIFAR10::channel_attention(torch::Tensor x) {
    auto input = x;
    x = global_pool->forward(x);
    x = x.reshape({x.size(0), 512});
    x = torch::relu(attention_fc1->forward(x));
    x = torch::sigmoid(attention_fc2->forward(x));
    x = x.reshape({-1, 512, 1, 1});
    return input * x;
}

torch::Tensor CNN_CIFAR10::forward(torch::Tensor x) {
    try{
        torch::Tensor residual = x;
        if (x.device() != m_device) {
            x = x.to(m_device);
            residual = residual.to(m_device);
        }

        x = torch::relu(bn1_1->forward(conv1_1->forward(x)));
        x = torch::relu(bn1_2->forward(conv1_2->forward(x)));
        x = torch::max_pool2d(x, 2);

        residual = x;
        residual = residual_bn1->forward(residual1->forward(residual));
        x = torch::relu(bn2_1->forward(conv2_1->forward(x)));
        x = torch::relu(bn2_2->forward(conv2_2->forward(x)));
        x = torch::relu(bn2_3->forward(conv2_3->forward(x)) + residual);
        x = spatial_dropout->forward(x);

        torch::Tensor residual3 = x;
        residual3 = residual_bn2->forward(residual2->forward(residual3));
        x = torch::relu(bn3_1->forward(conv3_1->forward(x)));
        x = torch::relu(bn3_2->forward(conv3_2->forward(x)));
        x = torch::relu(bn3_3->forward(conv3_3->forward(x)) + residual3);
        x = spatial_dropout->forward(x);

        x = torch::relu(bn4_1->forward(conv4_1->forward(x)));
        x = torch::relu(bn4_2->forward(conv4_2->forward(x)));
        x = channel_attention(x);

        x = x.reshape({x.size(0), -1});

        x = torch::relu(fc1->forward(x));
        x = dropout->forward(x);

        x = torch::relu(fc2->forward(x));
        x = dropout->forward(x);

        x = fc3->forward(x);

        return x;
    } catch(exception &e){
        cerr<<"ERROR forward "<<e.what()<<endl;
    }
    return x;
}

void CNN_CIFAR10::train(torch::Device device, int epochs, int batch_size) {
    string root = "../../data/cifar-10-batches-bin";
    if (!filesystem::exists(root)) {
        cerr << "错误：数据集路径不存在！" << endl;
        return;
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

    auto train_dataset = CIFAR10Dataset(root, CIFARDataset::Mode::kTrain).map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    cout << "train_dataset_size: " << train_dataset_size << endl;
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

    torch::optim::Adam optimizer(this->parameters(), /*lr=*/0.001);

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
}

double CNN_CIFAR10::test(torch::Device device, int batch_size) {
    string root = "../../data/cifar-10-batches-bin";
    if (!filesystem::exists(root)) {
        cerr << "错误：数据集路径不存在！" << endl;
        return 0.0;
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

    auto test_dataset = CIFAR10Dataset(root, CIFARDataset::Mode::kTest).map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    cout << "test_dataset_size: " << test_dataset_size << endl;
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);

    cout << "开始测试..." << endl;

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
    cout << "准确率: " << accuracy << endl;

    return accuracy;
}

int CNN_CIFAR10::predict(torch::Tensor input) {
    this->eval();
    torch::Tensor output = forward(input);
    if (m_device == torch::kCUDA) {
        output = output.to(torch::kCPU);
    }
    return output.argmax(1).item<int>();
}

void CNN_CIFAR10::save(torch::serialize::OutputArchive& archive) const {
    torch::nn::Module::save(archive);
    archive.write("device", m_device.type() == torch::kCUDA ? "cuda" : "cpu");
}

void CNN_CIFAR10::load(torch::serialize::InputArchive& archive) {
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

void CNN_CIFAR10::save(const string& path) {
    try {
        torch::serialize::OutputArchive archive;
        torch::nn::Module::save(archive);
        archive.write("device", m_device.type() == torch::kCUDA ? "cuda" : "cpu");
        archive.save_to(path);
        cout << "CNN_CIFAR10模型参数保存成功，路径：" << path << endl;
    } catch (const exception& e) {
        cerr << "保存模型参数失败：" << e.what() << endl;
        throw;
    }
}

void CNN_CIFAR10::load(const string& path) {
    try {
        if (!filesystem::exists(path)) {
            throw runtime_error("文件不存在：" + path);
        }
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        torch::nn::Module::load(archive);
        c10::IValue device_str;
        if (archive.try_read("device", device_str)) {
            m_device = (device_str == "cuda" && torch::cuda::is_available()) ?
                           torch::kCUDA : torch::kCPU;
            this->to(m_device);
        }
        cout << "CNN_CIFAR10模型参数加载成功，路径：" << path << endl;
    } catch (const exception& e) {
        cerr << "加载模型参数失败：" << e.what() << endl;
        throw;
    }
}

torch::Device CNN_CIFAR10::device() const {
    return m_device;
}

CNN_CIFAR100::CNN_CIFAR100(int num_classes, int input_channels) : m_device(torch::kCUDA) {
    try {
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 64, 3).padding(1)));
        bn1_1 = register_module("bn1_1", torch::nn::BatchNorm2d(64));
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        bn1_2 = register_module("bn1_2", torch::nn::BatchNorm2d(64));

        conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1).stride(2)));
        bn2_1 = register_module("bn2_1", torch::nn::BatchNorm2d(128));
        conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
        bn2_2 = register_module("bn2_2", torch::nn::BatchNorm2d(128));
        conv2_3 = register_module("conv2_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
        bn2_3 = register_module("bn2_3", torch::nn::BatchNorm2d(128));

        conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1).stride(2)));
        bn3_1 = register_module("bn3_1", torch::nn::BatchNorm2d(256));
        conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        bn3_2 = register_module("bn3_2", torch::nn::BatchNorm2d(256));
        conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        bn3_3 = register_module("bn3_3", torch::nn::BatchNorm2d(256));

        conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
        bn4_1 = register_module("bn4_1", torch::nn::BatchNorm2d(512));
        conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
        bn4_2 = register_module("bn4_2", torch::nn::BatchNorm2d(512));

        residual1 = register_module("residual1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1).stride(2)));
        residual_bn1 = register_module("residual_bn1", torch::nn::BatchNorm2d(128));
        residual2 = register_module("residual2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1).stride(2)));
        residual_bn2 = register_module("residual_bn2", torch::nn::BatchNorm2d(256));

        global_pool = register_module("global_pool", torch::nn::AdaptiveAvgPool2d(1));
        attention_fc1 = register_module("attention_fc1", torch::nn::Linear(512, 128));
        attention_fc2 = register_module("attention_fc2", torch::nn::Linear(128, 512));

        fc1 = register_module("fc1", torch::nn::Linear(512 * 4 * 4, 1024));
        fc2 = register_module("fc2", torch::nn::Linear(1024, 512));
        fc3 = register_module("fc3", torch::nn::Linear(512, num_classes));

        spatial_dropout = register_module("spatial_dropout", torch::nn::Dropout2d(0.1));
        dropout = register_module("dropout", torch::nn::Dropout(0.3));
    } catch (const exception& e) {
        cerr<<"ERROR CNN_CIFAR100"<<e.what()<<endl;
        throw;
    }

    if (torch::cuda::is_available()) {
        m_device = torch::kCUDA;
        cout << "CUDA is available! Using GPU auto." << endl;
    }
    else {
        m_device = torch::kCPU;
        cout << "CUDA not available. Using CPU auto." << endl;
    }

    this->to(m_device);
}

torch::Tensor CNN_CIFAR100::channel_attention(torch::Tensor x) {
    auto input = x;
    x = global_pool->forward(x);
    x = x.reshape({x.size(0), 512});
    x = torch::relu(attention_fc1->forward(x));
    x = torch::sigmoid(attention_fc2->forward(x));
    x = x.reshape({-1, 512, 1, 1});
    return input * x;
}

torch::Tensor CNN_CIFAR100::forward(torch::Tensor x) {
    try{
        auto residual = x;
        if (x.device() != m_device) {
            x = x.to(m_device);
            residual = residual.to(m_device);
        }

        x = torch::relu(bn1_1->forward(conv1_1->forward(x)));
        x = torch::relu(bn1_2->forward(conv1_2->forward(x)));
        x = torch::max_pool2d(x, 2);

        residual = x;
        residual = residual_bn1->forward(residual1->forward(residual));
        x = torch::relu(bn2_1->forward(conv2_1->forward(x)));
        x = torch::relu(bn2_2->forward(conv2_2->forward(x)));
        x = torch::relu(bn2_3->forward(conv2_3->forward(x)) + residual);
        x = spatial_dropout->forward(x);

        auto residual3 = x;
        residual3 = residual_bn2->forward(residual2->forward(residual3));
        x = torch::relu(bn3_1->forward(conv3_1->forward(x)));
        x = torch::relu(bn3_2->forward(conv3_2->forward(x)));
        x = torch::relu(bn3_3->forward(conv3_3->forward(x)) + residual3);
        x = spatial_dropout->forward(x);

        x = torch::relu(bn4_1->forward(conv4_1->forward(x)));
        x = torch::relu(bn4_2->forward(conv4_2->forward(x)));
        x = channel_attention(x);

        x = x.reshape({x.size(0), -1});

        x = torch::relu(fc1->forward(x));
        x = dropout->forward(x);

        x = torch::relu(fc2->forward(x));
        x = dropout->forward(x);

        x = fc3->forward(x);

        return x;
    }
    catch(exception &e){
        cerr<<"ERROR forward"<<e.what()<<endl;
    }
    return x;
}

void CNN_CIFAR100::train(torch::Device device, int epochs, int batch_size) {
    string root = "../../data/cifar-100-binary";
    if (!filesystem::exists(root)) {
        cerr << "错误：数据集路径不存在！" << endl;
        return;
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

    auto train_dataset = CIFAR100Dataset(root, CIFARDataset::Mode::kTrain).map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    cout << "train_dataset_size: " << train_dataset_size << endl;
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

    torch::optim::Adam optimizer(this->parameters(), /*lr=*/0.001);

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
}

double CNN_CIFAR100::test(torch::Device device, int batch_size) {
    string root = "../../data/cifar-100-binary";
    if (!filesystem::exists(root)) {
        cerr << "CNN_CIFAR100::test PATH FAIL!" << endl;
        return 0.0;
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

    auto test_dataset = CIFAR100Dataset(root, CIFARDataset::Mode::kTest).map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    cout << "test_dataset_size: " << test_dataset_size << endl;
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);

    cout << "开始测试..." << endl;

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
    cout << "准确率: " << accuracy << endl;

    return accuracy;
}

int CNN_CIFAR100::predict(torch::Tensor input) {
    this->eval();
    torch::Tensor output = forward(input);
    if (m_device == torch::kCUDA) {
        output = output.to(torch::kCPU);
    }
    return output.argmax(1).item<int>();
}

void CNN_CIFAR100::save(torch::serialize::OutputArchive& archive) const {
    torch::nn::Module::save(archive);
    archive.write("device", m_device.type() == torch::kCUDA ? "cuda" : "cpu");
}

void CNN_CIFAR100::load(torch::serialize::InputArchive& archive) {
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

void CNN_CIFAR100::save(const string& path) {
    try {
        torch::serialize::OutputArchive archive;
        torch::nn::Module::save(archive);
        archive.write("device", m_device.type() == torch::kCUDA ? "cuda" : "cpu");
        archive.save_to(path);
        cout << "CNN_CIFAR100模型参数保存成功，路径：" << path << endl;
    } catch (const exception& e) {
        cerr << "保存模型参数失败：" << e.what() << endl;
        throw;
    }
}

void CNN_CIFAR100::load(const string& path) {
    try {
        if (!filesystem::exists(path)) {
            throw runtime_error("文件不存在：" + path);
        }

        torch::serialize::InputArchive archive;
        archive.load_from(path);
        torch::nn::Module::load(archive);

        c10::IValue device_str;
        if (archive.try_read("device", device_str)) {
            m_device = (device_str == "cuda" && torch::cuda::is_available()) ?
                           torch::kCUDA : torch::kCPU;
            this->to(m_device);
        }
        cout << "CNN_CIFAR100模型参数加载成功，路径：" << path << endl;
    } catch (const exception& e) {
        cerr << "加载模型参数失败：" << e.what() << endl;
        throw;
    }
}

torch::Device CNN_CIFAR100::device() const {
    return m_device;
}

string CNN_CIFAR10::classify(int result) {
    const vector<string> CIFAR10_CLASSES = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    return CIFAR10_CLASSES[result];
}

string CNN_CIFAR100::classify(int result) {
    const vector<string> CIFAR100_CLASSES = {
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
        "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
        "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
        "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
        "worm"
    };
    return CIFAR100_CLASSES[result];
}
