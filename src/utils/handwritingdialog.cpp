// 数据结构课程设计 - [CNN开放题]
// 版权所有 (c) [2025] [爱玩的尼克]
// 根据 [MIT License] 许可证发布。详情请见项目根目录下的 LICENSE 文件。
#include "handwritingdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QMessageBox>

HandwritingCanvas::HandwritingCanvas(QWidget *parent)
    : QWidget(parent), m_canvasImage(280, 280, QImage::Format_RGB32) {
    setFixedSize(280, 280);
    m_canvasImage.fill(Qt::black);
}

void HandwritingCanvas::clear() {
    m_canvasImage.fill(Qt::black);
    update();
}

torch::Tensor HandwritingCanvas::getInputTensor() {
    cout << "getInputTensor" << endl;

    QImage scaledImage = m_canvasImage.scaled(28, 28, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    cv::Mat rgbaMat(scaledImage.height(), scaledImage.width(), CV_8UC4, const_cast<uchar*>(scaledImage.bits()), static_cast<int>(scaledImage.bytesPerLine()));

    cv::Mat grayMat;
    cv::cvtColor(rgbaMat, grayMat, cv::COLOR_RGBA2GRAY);

    cv::Mat normalizedMat;
    grayMat.convertTo(normalizedMat, CV_32F, 1.0 / 255.0);
    normalizedMat = (normalizedMat - 0.1307) / 0.3081;

    return torch::from_blob(normalizedMat.data, {1, 1, 28, 28}, torch::kFloat32).clone();
}

void HandwritingCanvas::mousePressEvent(QMouseEvent *e) {
    if (e->button() == Qt::LeftButton) {
        m_lastPos = e->pos();
    }
}

void HandwritingCanvas::mouseMoveEvent(QMouseEvent *e) {
    if (e->buttons() & Qt::LeftButton) {
        QPainter painter(&m_canvasImage);
        painter.setPen(QPen(Qt::white, 15, Qt::SolidLine, Qt::RoundCap));
        painter.drawLine(m_lastPos, e->pos());
        m_lastPos = e->pos();
        update();
    }
}

void HandwritingCanvas::paintEvent(QPaintEvent *e) {
    Q_UNUSED(e);
    QPainter painter(this);
    painter.drawImage(rect(), m_canvasImage);
}

HandwritingDialog::HandwritingDialog(CNN_MNIST *model, QWidget *parent)
    : QDialog(parent), m_model(model) {
    cout << "HandwritingDialog" << endl;
    setWindowTitle("手写板");
    setFixedSize(300, 350);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    m_canvas = new HandwritingCanvas();
    QHBoxLayout *buttonLayout = new QHBoxLayout();

    QPushButton *clearBtn = new QPushButton("清空");
    QPushButton *predictBtn = new QPushButton("识别");
    connect(clearBtn, &QPushButton::clicked, m_canvas, &HandwritingCanvas::clear);
    connect(predictBtn, &QPushButton::clicked, this, &HandwritingDialog::onPredictClicked);

    mainLayout->addWidget(m_canvas);
    mainLayout->addLayout(buttonLayout);
    buttonLayout->addWidget(clearBtn);
    buttonLayout->addWidget(predictBtn);
}

void HandwritingDialog::onPredictClicked() {
    cout << "onPredictClicked" << endl;
    torch::Tensor input = m_canvas->getInputTensor();
    input = input.to(m_model->device());
    int pred = m_model->predict(input);
    QMessageBox::information(this, "识别结果", QString("预测数字：%1").arg(pred));
}
