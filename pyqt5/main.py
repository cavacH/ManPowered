# -*- coding: utf-8 -*-

import os
import sys
caffe_root = '/home/hhd/manpowered/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe
import numpy as np
import argparse
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cv2
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel

    def detect(self, image, conf_thresh=0.8, topn=5):
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            result.append([xmin, ymin, xmax, ymax, label, score])
        return result

class CaffeClassification:
    def __init__(self, gpu_id, model_def, model_weight):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.net = caffe.Net(model_def, model_weight, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))

    def classify(self, image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        
        result = self.net.forward()['prob']

        return np.argmax(result[0])

class Ui_MainWindow(object):
    def __init__(self, args):
        self.detector = CaffeDetection(args.gpu_id, args.detection_prototxt, args.detection_model, args.detection_image_resize)
        self.classifier = CaffeClassification(args.gpu_id, args.classification_prototxt, args.classification_model)
        self.cap = cv2.VideoCapture(1)
        self.scale = 1.7 
        self.gap = 50
        self.begw = 50
        self.begh = 30
        self.time_gap = 0.1
        
        self.w = QtWidgets.QMainWindow()
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background,QtGui.QColor(173,216,230))
        self.w.setPalette(palette)
        self.w.setAutoFillBackground(True)
         
        self.timer= QTimer()
        self.setupUi()
        self.resize_widget(1280, 720)
        self.timer.timeout.connect(self.work)
        self.timer.start(self.time_gap)
    
    def work(self): 
        t1 = time.time()
        suc, frame = self.cap.read()
        result = self.detector.detect(frame)
        width = frame.shape[1]
        height = frame.shape[0]

        total = len(result)
        safe = 0
        unsafe = 0
       
        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
          
            xmin = max(0, int(round(xmin - (self.scale / 2.0 - 0.5) * (xmax - xmin))))
            xmax = min(width, int(round(xmax + (self.scale / 2.0 - 0.5) * (xmax - xmin))))
            ymin = max(0, int(round(ymin - (self.scale / 2.0 - 0.5) * (ymax - ymin))))
            ymax = min(height, int(round(ymax + (self.scale / 2.0 - 0.5) * (ymax - ymin))))
          
            sub_frame = cv2.resize(frame[ymin:ymax, xmin:xmax, :], (256, 256))[14:241, 14:241, :] 
            helmet_flag = self.classifier.classify(sub_frame)
            if helmet_flag:
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                safe += 1
            else:
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                unsafe += 1

        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        shot = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(shot))
        
        t2 = time.time()

        self.label_6.setText("%d" % total) 
        self.label_7.setText("%d" % safe)
        self.label_8.setText("%d" % unsafe)
        self.label_10.setText("%.3f" % (1.0 / (t2 - t1)))

    def show_window(self):
        self.w.show()
   
    def setupUi(self):
        self.w.setObjectName("MainWindow")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.w.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(self.w)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setScaledContents(True)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Segoe WP")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe WP")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe WP")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe WP")
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setTextFormat(QtCore.Qt.AutoText)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
         
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setText("")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setText("")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setText("")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_10.setFont(font)
        self.label_10.setText("")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
      
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.clicked_1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.clicked_2)
        
        self.w.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.w)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1007, 30))
        self.menubar.setObjectName("menubar")
        self.w.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.w)
        self.statusbar.setObjectName("statusbar")
        self.w.setStatusBar(self.statusbar)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.w)

    def resize_widget(self, width, height):
        self.buttonw = width / 5
        self.buttonh = self.gap
        self.labelh = height / 10
        self.labelw = self.labelh * 2
        hori_gap = (width - self.buttonw * 2) / 3
        vertical_gap = (height - self.labelh * 4) / 5 

        self.w.resize(self.begw + width + self.gap + self.labelw + self.gap + self.labelw + self.gap, self.begh + height + self.gap + self.buttonh + self.gap)
        self.label.setGeometry(QtCore.QRect(self.begw, self.begh, width, height))
        
        self.pushButton.setGeometry(QtCore.QRect(self.begw + hori_gap, self.begh + height + self.gap, self.buttonw, self.buttonh))
        self.pushButton_2.setGeometry(QtCore.QRect(self.begw + hori_gap + self.buttonw + hori_gap, self.begh + height + self.gap, self.buttonw, self.buttonh))
        
        self.label_3.setGeometry(QtCore.QRect(self.begw + width + self.gap, self.begh + vertical_gap, self.labelw, self.labelh))
        self.label_4.setGeometry(QtCore.QRect(self.begw + width + self.gap, self.begh + vertical_gap + self.labelh + vertical_gap, self.labelw, self.labelh))
        self.label_5.setGeometry(QtCore.QRect(self.begw + width + self.gap, self.begh + vertical_gap + (self.labelh + vertical_gap) * 2, self.labelw, self.labelh)) 
        self.label_9.setGeometry(QtCore.QRect(self.begw + width + self.gap, self.begh + vertical_gap + (self.labelh + vertical_gap) * 3, self.labelw, self.labelh))
        
        self.label_6.setGeometry(QtCore.QRect(self.begw + width + self.gap + self.labelw + self.gap, self.begh + vertical_gap, self.labelw, self.labelh))
        self.label_7.setGeometry(QtCore.QRect(self.begw + width + self.gap + self.labelw + self.gap, self.begh + vertical_gap + self.labelh + vertical_gap, self.labelw, self.labelh))
        self.label_8.setGeometry(QtCore.QRect(self.begw + width + self.gap + self.labelw + self.gap, self.begh + vertical_gap + (self.labelh + vertical_gap) * 2, self.labelw, self.labelh)) 
        self.label_10.setGeometry(QtCore.QRect(self.begw + width + self.gap + self.labelw + self.gap, self.begh + vertical_gap + (self.labelh + vertical_gap) * 3, self.labelw, self.labelh))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def clicked_1(self):
        self.resize_widget(1280, 720)

    def clicked_2(self):
        self.resize_widget(640, 480)
    
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.w.setWindowTitle(_translate("MainWindow", "Hardhat Detection"))
        self.label_3.setText(_translate("MainWindow", "Total"))
        self.label_4.setText(_translate("MainWindow", "Safe"))
        self.label_5.setText(_translate("MainWindow", "Unsafe"))
        self.label_9.setText(_translate("MainWindow", "FPS"))
        self.pushButton.setText(_translate("MainWindow", "1280x720"))
        self.pushButton_2.setText(_translate("MainWindow", "640x480"))

        file = self.menubar.addMenu("File")
        set = self.menubar.addMenu("Settings")
        help = self.menubar.addMenu("Help")
        nv = QAction('&New Video', self.centralwidget)
        oc = QAction('&Open Camera', self.centralwidget)
        file.addAction(nv)
        file.addAction(oc)
        res = set.addMenu("Camera Resolution")
        res1 = QAction('&1920*1080', self.centralwidget)
        res2 = QAction('&1280*720', self.centralwidget)
        res3 = QAction('&640*480', self.centralwidget)
        rl = QAction('Recording Location', self.centralwidget)
        sl = QAction('&Screenshot Location', self.centralwidget)
        res.addAction(res1)
        res.addAction(res2)
        res.addAction(res3)
        set.addAction(rl)
        set.addAction(sl)
        ins = QAction('&Instructions', self.centralwidget)
        ins.triggered.connect(self.show)
        help.addAction(ins)
        qss = QtCore.QFile('DarkOrange.qss')
        qss.open(QtCore.QFile.ReadOnly)
        styleSheet = qss.readAll()
        styleSheet = str(styleSheet, encoding='utf8')
        self.menubar.setStyleSheet(styleSheet)
        self.centralwidget.setStyleSheet(styleSheet)
    
    def show(self):
        #QMessageBox.about(self.centralwidget,"e","eee")
        dialog = Dialog(parent=self.centralwidget)
        if dialog.exec_():
            self.model.appendRow((
                QtGui.QStandardItem(dialog.name()),
                QtGui.QStandardItem(str(dialog.age())),
            ))
        dialog.destroy()

class Dialog(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.resize(540, 460)
        self.setWindowTitle("Document")
        grid = QGridLayout()
        t= QtWidgets.QTextBrowser(self)
        t.resize(540,460)
        t.setText("                   说明\n"
                  "New Video:导入新视频\nOpen Camera:打开摄像头\n"
                  "Camera Resolution:设置摄像头分辨率\n"
                  "Recording Location:录像保存位置\n"
                  "Screenshot Location:截图保存位置\n"
                  "分析：对视频进行安全帽检测\n"
                  "截图：对视频进行截图\n"
                  "录像：对视频进行录像\n"
                  "全屏：视频全屏显示\n"
                  "Total:视频中总人数\n"
                  "Safe:视频中戴安全帽的人数\n"
                  "Unsafe:视频中没戴安全帽的人数\n"
                  "如果视频中的人都戴了安全帽，右下角显示绿灯，否则显示红灯")
        t.setFont(QFont("Ubuntu Mono", 12))
        grid.addWidget(t,0,0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--detection_prototxt',
                        default=os.path.join(caffe_root, 'helmet_detection/detection.prototxt'))
    parser.add_argument('--detection_model',
                        default=os.path.join(caffe_root, 'helmet_detection/detection.caffemodel'))
    parser.add_argument('--classification_prototxt',
                        default=os.path.join(caffe_root, 'helmet_detection/classify.prototxt'))
    parser.add_argument('--classification_model',
                        default=os.path.join(caffe_root, 'helmet_detection/classify.caffemodel'))
    parser.add_argument('--detection_image_resize', default=300, type=int)
    
    return parser.parse_args()

def main(args):
    os.chdir(os.path.join(caffe_root, 'helmet_detection/'))   
    app = QtWidgets.QApplication(sys.argv)
    ex = Ui_MainWindow(args)
    ex.show_window()    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main(parse_args())

    
