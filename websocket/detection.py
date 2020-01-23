# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../backend/')

import caffe
import numpy as np

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean(
            'data', np.array([104, 117, 123]))

    def detect(self, image, conf_thresh=0.8, topn=5):
        self.net.blobs['data'].reshape(
            1, 3, self.image_resize, self.image_resize)

        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']

        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        top_indices = [i for i, conf in enumerate(
            det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i]
            ymin = top_ymin[i]
            xmax = top_xmax[i]
            ymax = top_ymax[i]
            score = top_conf[i]
            label = int(top_label_indices[i])
            result.append([xmin, ymin, xmax, ymax, label, score])
        return result


class CaffeClassification:
    def __init__(self, gpu_id, model_def, model_weight):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.net = caffe.Net(model_def, model_weight, caffe.TEST)
        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))

    def classify(self, image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess(
            'data', image)

        result = self.net.forward()['prob']

        return np.argmax(result[0])



