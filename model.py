import cv2
import numpy as np
import onnx
import onnxruntime as ort


class Yolov5ONNX(object):
    def __init__(self, onnx_path):
        """检查onnx模型并初始化onnx"""
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        else:
            print("Model correct")

        options = ort.SessionOptions()
        options.enable_profiling = True
        # self.onnx_session = ort.InferenceSession(onnx_path, sess_options=options,
        #                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.onnx_session = ort.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()  # ['images']
        self.output_name = self.get_output_name()  # ['output0']

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        """获取输出节点名称"""
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy

        return input_feed

    def inference(self, img):
        """
        onnx_session 推理
        """
        input_feed = self.get_input_feed(img)  # dict:{ input_name: input_value }
        pred = self.onnx_session.run(None, input_feed)[0]  # <class 'numpy.ndarray'>(1, 25200, 9)

        return pred

    def inference_frame(self, frame):
        """
        1. 将视频帧进行resize
        2. 图像转BGR2RGB和HWC2CHW（因为yolov5的onnx模型输入为 RGB：1 × 3 × 640 × 640）
        3. 图像归一化
        4. 图像增加维度
        5. onnx_session推理
        """
        or_img = cv2.resize(frame, (640, 640))
        img = or_img[:, :, ::-1].transpose((2, 0, 1))
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]

        return pred, or_img
