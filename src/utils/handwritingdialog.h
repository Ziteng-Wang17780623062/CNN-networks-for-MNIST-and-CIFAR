// ���ݽṹ�γ���� - [CNN������]
// ��Ȩ���� (c) [2025] [��������]
// ���� [MIT License] ���֤���������������Ŀ��Ŀ¼�µ� LICENSE �ļ���
#ifndef HANDWRITINGDIALOG_H
#define HANDWRITINGDIALOG_H

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <QDialog>
#include <QMouseEvent>
#include <QPainter>
#include "model_mnist.h"

class HandwritingCanvas : public QWidget {
    Q_OBJECT
public:
    explicit HandwritingCanvas(QWidget *parent = nullptr);
    void clear();
    torch::Tensor getInputTensor();
protected:
    void mousePressEvent(QMouseEvent *e) override;
    void mouseMoveEvent(QMouseEvent *e) override;
    void paintEvent(QPaintEvent *e) override;

private:
    QImage m_canvasImage;
    QPoint m_lastPos;
};

class HandwritingDialog : public QDialog {
    Q_OBJECT
public:
    explicit HandwritingDialog(CNN_MNIST *model, QWidget *parent = nullptr);

private slots:
    void onPredictClicked();

private:
    HandwritingCanvas *m_canvas;
    CNN_MNIST *m_model;
};

#endif // HANDWRITINGDIALOG_H
