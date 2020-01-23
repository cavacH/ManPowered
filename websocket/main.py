from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer
from base64 import b64decode, b64encode
import numpy as np
import detection
import cv2
import time

class ImageServer(WebSocket):

    counter = 0
    total_time = 0

    def detect_frame(self, cv2_frame):
        scale = 1.4
        bboxes = self.detector.detect(cv2_frame)
        width = cv2_frame.shape[1]
        height = cv2_frame.shape[0]

        for bbox in bboxes:
            xmin = int(round(bbox[0] * width))
            ymin = int(round(bbox[1] * height))
            xmax = int(round(bbox[2] * width))
            ymax = int(round(bbox[3] * height))
            xmin = max(
                0, int(round(xmin - (scale / 2.0 - 0.5) * (xmax - xmin))))
            xmax = min(width, int(
                round(xmax + (scale / 2.0 - 0.5) * (xmax - xmin))))
            ymin = max(
                0, int(round(ymin - (scale / 2.0 - 0.5) * (ymax - ymin))))
            ymax = min(height, int(
                round(ymax + (scale / 2.0 - 0.5) * (ymax - ymin))))

            sub_frame = cv2.resize(cv2_frame[ymin:ymax, xmin:xmax, :], (256, 256))[
                14:241, 14:241, :]
            if self.classifier.classify(sub_frame):
                cv2_frame = cv2.rectangle(
                    cv2_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            else:
                cv2_frame = cv2.rectangle(
                    cv2_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        return cv2_frame
    
    def handleMessage(self):
        t = time.time()
        data = self.data.split(',')
        img = b64decode(data[1])
        np_img = np.fromstring(img, np.uint8)
        cv2_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        cv2_img = self.detect_frame(cv2_img)
        retval, buffer = cv2.imencode('.jpg', cv2_img)
        buffer = b64encode(buffer)
        self.sendMessage(data[0]+','+buffer)
        self.total_time += time.time() - t
        self.counter += 1
        if self.counter is 100:
            print('avg time:', self.total_time / self.counter)
            self.total_time = 0
            self.counter = 0

    def handleConnected(self):
        print(self.address, 'connected')
        self.detector = detection.CaffeDetection( 0, '../model/detection.prototxt', '../model/detection.caffemodel', 300)
        self.classifier = detection.CaffeClassification(
    0, '../model/classify.prototxt', '../model/classify.caffemodel')

    def handleClose(self):
        print(self.address, 'closed')

server = SimpleWebSocketServer('', 8000, ImageServer)
server.serveforever()
