import cv2
import numpy as np
from model import Yolov5ONNX
from utils import xywh2xyxy, nms
import time


class CameraDetector:
    def __init__(self, onnx_path, classes, conf_thres=0.5, iou_thres=0.5):
        self.classes = classes
        self.onnx_path = onnx_path
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.model = self.load_model()

    def load_model(self):
        model = Yolov5ONNX(self.onnx_path)
        return model

    def filter_box(self, org_box):
        org_box = np.squeeze(org_box)

        conf = org_box[..., 4] > self.conf_thres
        box = org_box[conf == True]

        cls_cinf = box[..., 5:]
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))
        all_cls = list(set(cls))

        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []

            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])

            curr_cls_box = np.array(curr_cls_box)
            curr_cls_box = xywh2xyxy(curr_cls_box)
            curr_out_box = nms(curr_cls_box, self.iou_thres)

            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    def draw(self, image, box_data):
        try:
            scores = box_data[..., 4]
        except IndexError:
            print("当前画面无检测目标")
            return image
        boxes = box_data[..., :4].astype(np.int32)
        classes = box_data[..., 5].astype(np.int32)
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            # print('class: {}, score: {}'.format(self.classes[cl], score))
            # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.classes[cl], score),
                        (top, left),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        return image

    def camera_on(self):
        capture = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = capture.read()

            output, or_img = self.model.inference_frame(frame)
            outbox = self.filter_box(output)

            or_img = self.draw(or_img, outbox)
            cv2.imshow("Detection", or_img)

            frame_count += 1

            # FPS
            if frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print("FPS: {:.2f}".format(fps))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = 'weights/yolov5n_face_batch_1.onnx'
    CLASSES = ['face', 'eye', 'mouth']
    detector = CameraDetector(model_path, classes=CLASSES, conf_thres=0.45, iou_thres=0.45)
    detector.camera_on()
