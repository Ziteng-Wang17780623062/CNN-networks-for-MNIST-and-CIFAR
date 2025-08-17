// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "model_mnist.h"
#include "model_cifar.h"
#include "trainingthread.h"
#include "testthread.h"
#include "trainingthread_cifar.h"
#include "testthread_cifar.h"

#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QtWidgets>
#include <QLabel>

constexpr int BUTTON_WIDTH = 150;
constexpr int BUTTON_HEIGHT = 30;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
        setupUI();

        m_mnistModel = new CNN_MNIST(10, 1);
        m_cifar10Model = new CNN_CIFAR10(10, 3);
        m_cifar100Model = new CNN_CIFAR100(100, 3);

        m_trainingThread = new TrainingThread(m_mnistModel, this);
        m_trainingThreadCIFAR10 = new TrainingThreadCIFAR10(m_cifar10Model, this);
        m_trainingThreadCIFAR100 = new TrainingThreadCIFAR100(m_cifar100Model, this);

        m_testThread = new TestThread(m_mnistModel, torch::kCUDA, 100, this);
        m_testThreadCIFAR10 = new TestThreadCIFAR10(m_cifar10Model, torch::kCUDA, 100, this);
        m_testThreadCIFAR100 = new TestThreadCIFAR100(m_cifar100Model, torch::kCUDA, 100, this);

        setupConnections();
    }

public slots:
    void onTrainButtonClicked();
    void onTestButtonClicked();
    void onUseModelButtonClicked();
    void onShowDataButtonClicked();

    void onDeviceChanged(const QString &device);

    void onModelChanged(const QString &model);
    void onLoadModelClicked();
    void onSaveModelClicked();

    void onTrainingStarted();
    void onTrainingProgress(int progress);
    void onTrainingEpochCompleted(int total, int epoch, double loss, double accuracy);
    void onTrainingFinished();
    void onTrainingErrorOccurred(const QString &msg);

    void onTestStart();
    void onTestFinished(double acc);
    void onTestErrorOccurred(const QString &msg);

private:
    void setupUI();

    void setupConnections();

    QComboBox *modelComboBox;
    QComboBox *deviceComboBox;
    QLineEdit *epochsLineEdit;
    QLineEdit *batchesLineEdit;
    QPushButton *trainButton;
    QPushButton *testButton;
    QPushButton *useModelButton;
    QPushButton *clearLogButton;
    QPushButton *showDataButton;
    QPushButton *saveModelButton;
    QPushButton *loadModelButton;
    QLabel *statusLabel;
    QProgressBar *progressBar;
    QTextEdit *logTextEdit;

    QList<QList<int>> allEpochList;
    QList<QList<double>> allAccuracyHistory;
    QList<QColor> trainRecordColors;
    QList<double> currentAccuracy;
    QList<int> currentEpochs;

    QList<QList<int>> allEpochList_cifar10;
    QList<QList<double>> allAccuracyHistory_cifar10;
    QList<QColor> trainRecordColors_cifar10;
    QList<double> currentAccuracy_cifar10;
    QList<int> currentEpochs_cifar10;

    QList<QList<int>> allEpochList_cifar100;
    QList<QList<double>> allAccuracyHistory_cifar100;
    QList<QColor> trainRecordColors_cifar100;
    QList<double> currentAccuracy_cifar100;
    QList<int> currentEpochs_cifar100;

    TrainingThread *m_trainingThread;
    TrainingThreadCIFAR10 *m_trainingThreadCIFAR10;
    TrainingThreadCIFAR100 *m_trainingThreadCIFAR100;

    TestThread *m_testThread;
    TestThreadCIFAR10 *m_testThreadCIFAR10;
    TestThreadCIFAR100 *m_testThreadCIFAR100;

    CNN_MNIST *m_mnistModel;
    CNN_CIFAR10 *m_cifar10Model;
    CNN_CIFAR100 *m_cifar100Model;
};

#endif // MAINWINDOW_H
