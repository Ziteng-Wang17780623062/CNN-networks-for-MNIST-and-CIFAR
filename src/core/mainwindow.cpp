// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
#include "mainwindow.h"
#include "cifardataset.h"
#include "handwritingdialog.h"
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>

void MainWindow::setupUI() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    QLabel *titleLabel = new QLabel("CNN���ѧϰģ��ѵ�����Խ���", this);
    titleLabel->setFont(QFont("Arial", 16, QFont::Bold));
    titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(titleLabel);

    QFrame *line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    mainLayout->addWidget(line);

    QHBoxLayout *modelSelectLayout = new QHBoxLayout();
    QLabel *modelLabel = new QLabel("ѡ��ģ��:", this);
    modelComboBox = new QComboBox(this);
    modelComboBox->addItem("CNN_MNIST");
    modelComboBox->addItem("CNN_CIFAR10");
    modelComboBox->addItem("CNN_CIFAR100");
    modelComboBox->setMinimumWidth(200);

    saveModelButton = new QPushButton("����ģ�Ͳ���", this);
    saveModelButton->setMinimumWidth(120);
    loadModelButton = new QPushButton("����ģ�Ͳ���", this);
    loadModelButton->setMinimumWidth(120);

    modelSelectLayout->addWidget(modelLabel);
    modelSelectLayout->addWidget(modelComboBox);
    modelSelectLayout->addWidget(loadModelButton);
    modelSelectLayout->addWidget(saveModelButton);
    modelSelectLayout->addStretch();

    mainLayout->addLayout(modelSelectLayout);

    QHBoxLayout *deviceSelectLayout = new QHBoxLayout();
    QLabel *deviceLabel = new QLabel("�����豸:", this);
    deviceComboBox = new QComboBox(this);
    deviceComboBox->addItem("Intel Core i9 CPU");
    deviceComboBox->addItem("NVIDIA GEFORCE RTX 4090 LapTop");
    deviceComboBox->setMinimumWidth(300);

    deviceSelectLayout->addWidget(deviceLabel);
    deviceSelectLayout->addWidget(deviceComboBox);
    deviceSelectLayout->addStretch();

    mainLayout->addLayout(deviceSelectLayout);

    QHBoxLayout *paramLayout = new QHBoxLayout();
    QLabel *epochsLabel = new QLabel("ѵ������ (Epochs):", this);
    epochsLineEdit = new QLineEdit(this);
    epochsLineEdit->setPlaceholderText("����ѵ������");
    epochsLineEdit->setValidator(new QIntValidator(1, 1000, this));
    epochsLineEdit->setMaximumWidth(100);

    paramLayout->addWidget(epochsLabel);
    paramLayout->addWidget(epochsLineEdit);
    paramLayout->addStretch();

    mainLayout->addLayout(paramLayout);

    QHBoxLayout *paramLayout1 = new QHBoxLayout();
    QLabel *batchesLabel = new QLabel("�������� (Batches):", this);
    batchesLineEdit = new QLineEdit(this);
    batchesLineEdit->setPlaceholderText("������������");
    batchesLineEdit->setValidator(new QIntValidator(1, 1000, this));
    batchesLineEdit->setMaximumWidth(100);

    paramLayout1->addWidget(batchesLabel);
    paramLayout1->addWidget(batchesLineEdit);
    paramLayout1->addStretch();

    mainLayout->addLayout(paramLayout1);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    trainButton = new QPushButton("ѵ��ģ��", this);
    testButton = new QPushButton("����ģ��", this);
    showDataButton = new QPushButton("��ʾѵ������", this);
    useModelButton = new QPushButton("ʹ��ģ��");

    buttonLayout->addWidget(trainButton);
    buttonLayout->addWidget(testButton);
    buttonLayout->addWidget(useModelButton);
    buttonLayout->addWidget(showDataButton);

    mainLayout->addLayout(buttonLayout);

    statusLabel = new QLabel("����", this);
    mainLayout->addWidget(statusLabel);

    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setStyle(QStyleFactory::create("Fusion"));
    mainLayout->addWidget(progressBar);

    logTextEdit = new QTextEdit(this);
    logTextEdit->setReadOnly(true);
    logTextEdit->setPlaceholderText("ѵ����־����ʾ������...");
    mainLayout->addWidget(logTextEdit);

    clearLogButton = new QPushButton("�����־", this);
    clearLogButton->setFixedWidth(logTextEdit->width());
    connect(clearLogButton, &QPushButton::clicked, logTextEdit, &QTextEdit::clear);
    mainLayout->addWidget(clearLogButton);

    setWindowTitle("����CNN�����ѧϰ����ѵ������ƽ̨");
    setMinimumSize(800, 600);
    resize(900, 700);
}

void MainWindow::setupConnections() {
    connect(trainButton, &QPushButton::clicked, this, &MainWindow::onTrainButtonClicked);
    connect(testButton, &QPushButton::clicked, this, &MainWindow::onTestButtonClicked);
    connect(useModelButton, &QPushButton::clicked, this, &MainWindow::onUseModelButtonClicked);
    connect(showDataButton, &QPushButton::clicked, this, &MainWindow::onShowDataButtonClicked);

    connect(deviceComboBox, &QComboBox::currentTextChanged, this, &MainWindow::onDeviceChanged);

    connect(modelComboBox, &QComboBox::currentTextChanged, this, &MainWindow::onModelChanged);

    connect(loadModelButton, &QPushButton::clicked, this, &MainWindow::onLoadModelClicked);
    connect(saveModelButton, &QPushButton::clicked, this, &MainWindow::onSaveModelClicked);

    connect(m_trainingThread, &TrainingThread::trainingStarted, this, &MainWindow::onTrainingStarted);
    connect(m_trainingThread, &TrainingThread::progressUpdated, this, &MainWindow::onTrainingProgress);
    connect(m_trainingThread, &TrainingThread::epochCompleted, this, &MainWindow::onTrainingEpochCompleted);
    connect(m_trainingThread, &TrainingThread::trainingFinished, this, &MainWindow::onTrainingFinished);
    connect(m_trainingThread, &TrainingThread::errorOccurred, this, &MainWindow::onTrainingErrorOccurred);

    connect(m_trainingThreadCIFAR10, &TrainingThreadCIFAR10::trainingStarted, this, &MainWindow::onTrainingStarted);
    connect(m_trainingThreadCIFAR10, &TrainingThreadCIFAR10::progressUpdated, this, &MainWindow::onTrainingProgress);
    connect(m_trainingThreadCIFAR10, &TrainingThreadCIFAR10::epochCompleted, this, &MainWindow::onTrainingEpochCompleted);
    connect(m_trainingThreadCIFAR10, &TrainingThreadCIFAR10::trainingFinished, this, &MainWindow::onTrainingFinished);
    connect(m_trainingThreadCIFAR10, &TrainingThreadCIFAR10::errorOccurred, this, &MainWindow::onTrainingErrorOccurred);

    connect(m_trainingThreadCIFAR100, &TrainingThreadCIFAR100::trainingStarted, this, &MainWindow::onTrainingStarted);
    connect(m_trainingThreadCIFAR100, &TrainingThreadCIFAR100::progressUpdated, this, &MainWindow::onTrainingProgress);
    connect(m_trainingThreadCIFAR100, &TrainingThreadCIFAR100::epochCompleted, this, &MainWindow::onTrainingEpochCompleted);
    connect(m_trainingThreadCIFAR100, &TrainingThreadCIFAR100::trainingFinished, this, &MainWindow::onTrainingFinished);
    connect(m_trainingThreadCIFAR100, &TrainingThreadCIFAR100::errorOccurred, this, &MainWindow::onTrainingErrorOccurred);

    connect(m_testThread, &TestThread::testStarted, this, &MainWindow::onTestStart);
    connect(m_testThread, &TestThread::testFinished, this, &MainWindow::onTestFinished);
    connect(m_testThread, &TestThread::errorOccurred, this, &MainWindow::onTestErrorOccurred);

    connect(m_testThreadCIFAR10, &TestThreadCIFAR10::testStarted, this, &MainWindow::onTestStart);
    connect(m_testThreadCIFAR10, &TestThreadCIFAR10::testFinished, this, &MainWindow::onTestFinished);
    connect(m_testThreadCIFAR10, &TestThreadCIFAR10::errorOccurred, this, &MainWindow::onTestErrorOccurred);

    connect(m_testThreadCIFAR100, &TestThreadCIFAR100::testStarted, this, &MainWindow::onTestStart);
    connect(m_testThreadCIFAR100, &TestThreadCIFAR100::testFinished, this, &MainWindow::onTestFinished);
    connect(m_testThreadCIFAR100, &TestThreadCIFAR100::errorOccurred, this, &MainWindow::onTestErrorOccurred);
}

void MainWindow::onTrainButtonClicked() {
    QString modelName = modelComboBox->currentText();
    QString device = deviceComboBox->currentText();
    QString epochsText = epochsLineEdit->text();
    QString batchesText = batchesLineEdit->text();
    if (modelName.isEmpty()) {
        QMessageBox::warning(this, "����", "��ѡ��ģ��");
        return;
    }
    bool ok;
    int epochs = epochsText.toInt(&ok);
    if (!ok || epochs <= 0) {
        QMessageBox::warning(this, "����", "��������Ч��ѵ����������1��");
        return;
    }
    bool ok1;
    int batches = batchesText.toInt(&ok1);
    if (!ok1 || batches <= 0) {
        QMessageBox::warning(this, "����", "��������Ч��������������1��");
        return;
    }
    logTextEdit->append(QString("=== ��ʼѵ��ģ��: %1 ===").arg(modelName));
    currentAccuracy.clear();
    currentEpochs.clear();
    if (modelName == "CNN_MNIST") {
        m_trainingThread->setParameters(modelName, device, epochs, batches);
        m_trainingThread->start();
    }
    else if (modelName == "CNN_CIFAR10") {
        m_trainingThreadCIFAR10->setParameters(modelName, device, epochs, batches);
        m_trainingThreadCIFAR10->start();
    }
    else if (modelName == "CNN_CIFAR100") {
        m_trainingThreadCIFAR100->setParameters(modelName, device, epochs, batches);
        m_trainingThreadCIFAR100->start();
    }
}

void MainWindow::onTestButtonClicked() {
    QString device = deviceComboBox->currentText();
    QString batchesText = batchesLineEdit->text();
    QString modelName = modelComboBox->currentText();
    bool ok1;
    int batches = batchesText.toInt(&ok1);
    if (!ok1 || batches <= 0) {
        QMessageBox::warning(this, "����", "��������Ч��������������1��");
        return;
    }
    if (modelName.isEmpty()) {
        QMessageBox::warning(this, "���Դ���", "��ѡ��һ��ģ��");
        return;
    }
    statusLabel->setText("������...");
    logTextEdit->append(QString("=== ��ʼ����ģ��: %1 ===").arg(modelName));

    if (modelName == "CNN_MNIST") {
        m_testThread->setParameters(m_mnistModel, device, batches);
        m_testThread->start();
    }
    else if (modelName == "CNN_CIFAR10") {
        m_testThreadCIFAR10->setParameters(m_cifar10Model, device, batches);
        m_testThreadCIFAR10->start();
    }
    else if (modelName == "CNN_CIFAR100") {
        m_testThreadCIFAR100->setParameters(m_cifar100Model, device, batches);
        m_testThreadCIFAR100->start();
    }
}

void MainWindow::onUseModelButtonClicked() {
    QString modelName = modelComboBox->currentText();
    if (modelName.isEmpty()) {
        QMessageBox::warning(this, "����", "��ѡ��ģ��");
        return;
    }
    logTextEdit->append(QString("=== ��ʼʹ�� %1 ģ�ʹ���ͼƬ ===").arg(modelName));
    if (modelName == "CNN_MNIST") {
        HandwritingDialog dialog(m_mnistModel, this);
        dialog.exec();
    }
    else {
        QString filePath = QFileDialog::getOpenFileName(this, "ѡ��ͼƬ", "D:\\MyProject\\QtProject\\CNN\\pictures", "Image Files (*.png *.jpg *.jpeg)");
        if (filePath.isEmpty()) {
            logTextEdit->append("�û�ȡ��ѡ��ͼƬ");
            return;
        }
        try {
            cv::Mat image = cv::imread(filePath.toStdString());
            if (image.empty()) {
                throw runtime_error("�޷���ȡͼƬ�ļ�: " + filePath.toStdString());
            }
            torch::Tensor inputTensor;
            if (modelName == "CNN_CIFAR10") {
                inputTensor = CIFARDataset::fromCVImage(image, CIFARDataset::Options());
                int result = m_cifar10Model->predict(inputTensor);
                QString result_str = QString::fromStdString(CNN_CIFAR10::classify(result));
                QMessageBox::information(this, "Ԥ����", QString("CIFAR10ģ��Ԥ����: %1").arg(result_str));
            }
            else if (modelName == "CNN_CIFAR100") {
                inputTensor = CIFARDataset::fromCVImage(image, CIFARDataset::Options());
                int result = m_cifar100Model->predict(inputTensor);
                QString result_str = QString::fromStdString(CNN_CIFAR100::classify(result));
                QMessageBox::information(this, "Ԥ����", QString("CIFAR100ģ��Ԥ����: %1").arg(result_str));
            }
            logTextEdit->append(QString("�ɹ�ʹ�� %1 ģ�ʹ���ͼƬ��·����%2").arg(modelName).arg(filePath));
        } catch (const exception& e) {
            logTextEdit->append(QString("����ʧ�ܣ�%1").arg(e.what()));
        }
    }
}

void MainWindow::onShowDataButtonClicked() {
    QString modelName = modelComboBox->currentText();
    if (modelName == "CNN_MNIST" && allAccuracyHistory.isEmpty()) {
        QMessageBox::information(this, "��ʾ", "����ʷѵ������");
        return;
    }
    else if (modelName == "CNN_CIFAR10" && allAccuracyHistory_cifar10.isEmpty()) {
        QMessageBox::information(this, "��ʾ", "����ʷѵ������");
        return;
    }
    else if (modelName == "CNN_CIFAR100" && allAccuracyHistory_cifar100.isEmpty()) {
        QMessageBox::information(this, "��ʾ", "����ʷѵ������");
        return;
    }

    QDialog *dialog = new QDialog(this);
    dialog->setWindowTitle("ѵ������");
    dialog->resize(800, 500);

    QVBoxLayout *mainLayout = new QVBoxLayout(dialog);
    QChart *chart = new QChart();
    chart->setTitle("����ѵ�����ȶԱ�");
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    // ��հ�ť
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    QPushButton *clearButton = new QPushButton("�������ѵ������");
    connect(clearButton, &QPushButton::clicked, this, [this, dialog, chart]() {
        allAccuracyHistory.clear();
        allEpochList.clear();
        trainRecordColors.clear();
        allAccuracyHistory_cifar10.clear();
        allEpochList_cifar10.clear();
        trainRecordColors_cifar10.clear();
        allAccuracyHistory_cifar100.clear();
        allEpochList_cifar100.clear();
        trainRecordColors_cifar100.clear();
        chart->removeAllSeries();
        dialog->close();
    });
    buttonLayout->addStretch();
    buttonLayout->addWidget(clearButton);

    if (modelName == "CNN_MNIST") {
        for (int recordIdx = 0; recordIdx < allAccuracyHistory.size(); ++recordIdx) {
            const QList<int>& epochs = allEpochList[recordIdx];
            const QList<double>& accuracies = allAccuracyHistory[recordIdx];

            QColor color;
            if (recordIdx < trainRecordColors.size()) {
                color = trainRecordColors[recordIdx];
            }
            else {
                color = Qt::red;
                qDebug() << "���棺��ɫ�б��Ȳ��㣬ʹ��Ĭ��Qt::red";
            }

            QLineSeries *lineSeries = new QLineSeries();
            lineSeries->setName(QString("ѵ����¼ %1").arg(recordIdx + 1));
            lineSeries->setPen(QPen(color, 2));

            QScatterSeries *scatterSeries = new QScatterSeries();
            scatterSeries->setMarkerSize(8);
            scatterSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
            scatterSeries->setColor(color);
            scatterSeries->setBorderColor(Qt::white);

            for (int i = 0; i < epochs.size(); ++i) {
                qDebug() << "��ӵ㣺epoch=" << epochs[i] << " accuracy=" << accuracies[i];
                lineSeries->append(epochs[i], accuracies[i]);
                scatterSeries->append(epochs[i], accuracies[i]);
            }

            chart->addSeries(lineSeries);
            chart->addSeries(scatterSeries);
        }
    }
    else if (modelName == "CNN_CIFAR10") {
        for (int recordIdx = 0; recordIdx < allAccuracyHistory_cifar10.size(); ++recordIdx) {
            const QList<int>& epochs = allEpochList_cifar10[recordIdx];
            const QList<double>& accuracies = allAccuracyHistory_cifar10[recordIdx];
            if (epochs.isEmpty() || accuracies.isEmpty()) continue;

            QColor color;
            if (recordIdx < trainRecordColors_cifar10.size()) {
                color = trainRecordColors_cifar10[recordIdx];
            }
            else {
                color = Qt::blue;
                qDebug() << "���棺��ɫ�б��Ȳ��㣬ʹ��Ĭ��Qt::blue";
            }

            QLineSeries *lineSeries = new QLineSeries();
            lineSeries->setName(QString("CIFAR10 ѵ����¼ %1").arg(recordIdx + 1));
            lineSeries->setPen(QPen(color, 2));

            QScatterSeries *scatterSeries = new QScatterSeries();
            scatterSeries->setMarkerSize(8);
            scatterSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
            scatterSeries->setColor(color);
            scatterSeries->setBorderColor(Qt::white);

            for (int i = 0; i < epochs.size(); ++i) {
                lineSeries->append(epochs[i], accuracies[i]);
                scatterSeries->append(epochs[i], accuracies[i]);
            }

            chart->addSeries(lineSeries);
            chart->addSeries(scatterSeries);
        }
    }
    else if (modelName == "CNN_CIFAR100") {
        for (int recordIdx = 0; recordIdx < allAccuracyHistory_cifar100.size(); ++recordIdx) {
            const QList<int>& epochs = allEpochList_cifar100[recordIdx];
            const QList<double>& accuracies = allAccuracyHistory_cifar100[recordIdx];
            if (epochs.isEmpty() || accuracies.isEmpty()) continue;

            QColor color;
            if (recordIdx < trainRecordColors_cifar10.size()) {
                color = trainRecordColors_cifar10[recordIdx];
            }
            else {
                color = Qt::green;
                qDebug() << "���棺��ɫ�б��Ȳ��㣬ʹ��Ĭ��Qt::green";
            }

            QLineSeries *lineSeries = new QLineSeries();
            lineSeries->setName(QString("CIFAR100 ѵ����¼ %1").arg(recordIdx + 1));
            lineSeries->setPen(QPen(color, 2));

            QScatterSeries *scatterSeries = new QScatterSeries();
            scatterSeries->setMarkerSize(8);
            scatterSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
            scatterSeries->setColor(color);
            scatterSeries->setBorderColor(Qt::white);

            for (int i = 0; i < epochs.size(); ++i) {
                lineSeries->append(epochs[i], accuracies[i]);
                scatterSeries->append(epochs[i], accuracies[i]);
            }

            chart->addSeries(lineSeries);
            chart->addSeries(scatterSeries);
        }
    }

    chart->createDefaultAxes();
    chart->axes(Qt::Horizontal).first()->setTitleText("Epoch");
    chart->axes(Qt::Vertical).first()->setTitleText("׼ȷ��(%)");
    chart->axes(Qt::Vertical).first()->setRange(0, 100);

    mainLayout->addLayout(buttonLayout);
    mainLayout->addWidget(chartView);
    dialog->show();
}

void MainWindow::onDeviceChanged(const QString &device) {
    statusLabel->setText(QString("�豸���л���: %1").arg(device));
    logTextEdit->append(QString("�豸����Ϊ: %1").arg(device));
}

void MainWindow::onModelChanged(const QString &model) {
    statusLabel->setText(QString("ģ�����л���: %1").arg(model));
    logTextEdit->append(QString("ģ������Ϊ: %1").arg(model));
}


void MainWindow::onSaveModelClicked() {
    QString modelName = modelComboBox->currentText();
    QString fileFilter = "PyTorch Model (*.pt);;ONNX Model (*.onnx)";
    QString savePath = QFileDialog::getSaveFileName(this, "����ģ��", "D:\\MyProject\\QtProject\\CNN\\models", fileFilter);

    if (savePath.isEmpty()) {
        return;
    }

    torch::Tensor dummy_input;
    if (modelName == "CNN_MNIST") {
        dummy_input = torch::randn({1, 1, 28, 28});
        if (savePath.endsWith(".pt")) {
            m_mnistModel->save(savePath.toStdString());
            QMessageBox::information(this, "����ɹ�", "ģ���� .pt ��ʽ����ɹ���");
        } else if (savePath.endsWith(".onnx")) {
            try {
                //m_mnistModel->saveAsONNX(savePath.toStdString(), dummy_input);
                QMessageBox::information(this, "����ɹ�", "ģ���� .onnx ��ʽ����ɹ���");
            } catch (const std::exception& e) {
                QMessageBox::critical(this, "����ʧ��", QString("����ΪONNX��ʽʧ��: %1").arg(e.what()));
            }
        }
    } else if (modelName == "CNN_CIFAR10") {
        dummy_input = torch::randn({1, 3, 32, 32});
        if (savePath.endsWith(".pt")) {
            m_cifar10Model->save(savePath.toStdString());
            QMessageBox::information(this, "����ɹ�", "ģ���� .pt ��ʽ����ɹ���");
        } else if (savePath.endsWith(".onnx")) {
            try {
                //m_cifar10Model->saveAsONNX(savePath.toStdString(), dummy_input);
                QMessageBox::information(this, "����ɹ�", "ģ���� .onnx ��ʽ����ɹ���");
            } catch (const std::exception& e) {
                QMessageBox::critical(this, "����ʧ��", QString("����ΪONNX��ʽʧ��: %1").arg(e.what()));
            }
        }
    } else if (modelName == "CNN_CIFAR100") {
        dummy_input = torch::randn({1, 3, 32, 32});
        if (savePath.endsWith(".pt")) {
            m_cifar100Model->save(savePath.toStdString());
            QMessageBox::information(this, "����ɹ�", "ģ���� .pt ��ʽ����ɹ���");
        } else if (savePath.endsWith(".onnx")) {
            try {
                //m_cifar100Model->saveAsONNX(savePath.toStdString(), dummy_input);
                QMessageBox::information(this, "����ɹ�", "ģ���� .onnx ��ʽ����ɹ���");
            } catch (const std::exception& e) {
                QMessageBox::critical(this, "����ʧ��", QString("����ΪONNX��ʽʧ��: %1").arg(e.what()));
            }
        }
    }
}

void MainWindow::onLoadModelClicked() {
    QString modelName = modelComboBox->currentText();
    if (modelName.isEmpty()) {
        QMessageBox::warning(this, "����", "��ѡ��ģ��");
        return;
    }

    QString loadPath = QFileDialog::getOpenFileName(this, "����ģ�Ͳ���", "D:\\MyProject\\QtProject\\CNN\\models", "Model Files (*.pt)");
    if (loadPath.isEmpty()) {
        logTextEdit->append("�û�ȡ������ģ�Ͳ���");
        return;
    }

    logTextEdit->append(QString("=== ��ʼ���� %1 ģ�Ͳ��� ===").arg(modelName));
    try {
        if (modelName == "CNN_MNIST") {
            m_mnistModel->load(loadPath.toStdString());
        }
        else if (modelName == "CNN_CIFAR10") {
            m_cifar10Model->load(loadPath.toStdString());
        }
        else if (modelName == "CNN_CIFAR100") {
            m_cifar100Model->load(loadPath.toStdString());
        }
        logTextEdit->append(QString("�ɹ����� %1 ģ�Ͳ�����·����%2").arg(modelName).arg(loadPath));
    } catch (const exception& e) {
        logTextEdit->append(QString("����ʧ�ܣ�%1").arg(e.what()));
    }
}

void MainWindow::onTrainingStarted() {
    progressBar->setValue(0);
    statusLabel->setText("ѵ����...");
    trainButton->setEnabled(false);
    testButton->setEnabled(false);
    useModelButton->setEnabled(false);
    modelComboBox->setEnabled(false);
}

void MainWindow::onTrainingProgress(int progress) {
    progressBar->setValue(progress);
}

void MainWindow::onTrainingEpochCompleted(int total, int epoch, double loss, double accuracy) {
    logTextEdit->append(QString("Epoch %1/%2: Loss = %3, Accuracy = %4%").arg(epoch).arg(total).arg(loss, 0, 'f', 4).arg(accuracy * 100.0, 0, 'f', 2));
    int progress = 100 * (epoch) / total;
    progressBar->setValue(progress);
    QString modelName = modelComboBox->currentText();
    if (modelName == "CNN_MNIST") {
        currentAccuracy.append(accuracy * 100.0);
        currentEpochs.append(epoch);
    }
    else if (modelName == "CNN_CIFAR10") {
        currentAccuracy_cifar10.append(accuracy * 100.0);
        currentEpochs_cifar10.append(epoch);
    }
    else if (modelName == "CNN_CIFAR100") {
        currentAccuracy_cifar100.append(accuracy * 100.0);
        currentEpochs_cifar100.append(epoch);
    }
}

void MainWindow::onTrainingFinished() {
    statusLabel->setText("ѵ�����");
    progressBar->setValue(100);
    trainButton->setEnabled(true);
    testButton->setEnabled(true);
    useModelButton->setEnabled(true);
    modelComboBox->setEnabled(true);
    logTextEdit->append("=== ѵ����� ===");

    QString modelName = modelComboBox->currentText();
    if (modelName == "CNN_MNIST") {
        if (!this->currentEpochs.isEmpty() && !this->currentAccuracy.isEmpty()) {
            allEpochList.append(this->currentEpochs);
            allAccuracyHistory.append(this->currentAccuracy);

            QColor newColor = QColor::fromHsvF(fmod(trainRecordColors.size() * 0.15, 1.0), 0.7, 0.9);
            trainRecordColors.append(newColor);
        }

        currentEpochs.clear();
        currentAccuracy.clear();
    }
    else if (modelName == "CNN_CIFAR10") {
        if (!this->currentEpochs_cifar10.isEmpty() && !this->currentAccuracy_cifar10.isEmpty()) {
            allEpochList_cifar10.append(this->currentEpochs_cifar10);
            allAccuracyHistory_cifar10.append(this->currentAccuracy_cifar10);

            QColor newColor = QColor::fromHsvF(fmod(trainRecordColors_cifar10.size() * 0.333, 1.0), 0.7, 0.9);
            trainRecordColors_cifar10.append(newColor);
        }

        currentEpochs_cifar10.clear();
        currentAccuracy_cifar10.clear();
    }
    else if (modelName == "CNN_CIFAR100") {
        if (!this->currentEpochs_cifar100.isEmpty() && !this->currentAccuracy_cifar100.isEmpty()) {
            allEpochList_cifar100.append(this->currentEpochs_cifar100);
            allAccuracyHistory_cifar100.append(this->currentAccuracy_cifar100);

            QColor newColor = QColor::fromHsvF(fmod(trainRecordColors_cifar100.size() * 0.667, 1.0), 0.7, 0.9);
            trainRecordColors_cifar100.append(newColor);
        }

        currentEpochs_cifar100.clear();
        currentAccuracy_cifar100.clear();
    }
}

void MainWindow::onTrainingErrorOccurred(const QString &msg) {
    QMessageBox::critical(this, "TestError", msg);
    statusLabel->setText("����");
    trainButton->setEnabled(true);
    testButton->setEnabled(true);
    useModelButton->setEnabled(true);
    modelComboBox->setEnabled(true);
}

void MainWindow::onTestStart() {
    statusLabel->setText("������...");
    trainButton->setEnabled(true);
    testButton->setEnabled(true);
    useModelButton->setEnabled(true);
    modelComboBox->setEnabled(true);
}

void MainWindow::onTestFinished(double acc) {
    statusLabel->setText(QString("������ɣ����ȣ� %1 %2").arg(acc * 100.0, 0, 'f', 2).arg('%'));
    trainButton->setEnabled(true);
    testButton->setEnabled(true);
    useModelButton->setEnabled(true);
    modelComboBox->setEnabled(true);
    logTextEdit->append(QString("Test finished, accuracy: %1 %2").arg(acc * 100.0, 0, 'f', 2).arg('%'));
    logTextEdit->append("=== ������� ===");
}

void MainWindow::onTestErrorOccurred(const QString &msg) {
    QMessageBox::critical(this, "TestError", msg);
    statusLabel->setText("����ʧ��");
    trainButton->setEnabled(true);
    testButton->setEnabled(true);
    useModelButton->setEnabled(true);
    modelComboBox->setEnabled(true);
}
