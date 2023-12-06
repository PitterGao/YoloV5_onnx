import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from utils import images_tensor_to_numpy
from utils.dataloader import images_numpy_to_tensor

class YOLODetector:
    def __init__(self, model_path, classes, o_device):
        self.device = o_device
        self.classes = classes
        self.model = attempt_load(model_path).to(self.device)
        self.model.requires_grad_(False)

    def detect(self, image_tensor):
        output = self.model(image_tensor)
        preds = non_max_suppression(output, 0.45, 0.45)
        return preds

    def draw(self, image_tensor, box_data):
        box_data = box_data.cpu().numpy()
        image = images_tensor_to_numpy(image_tensor)
        boxes = box_data[..., :4].astype(np.int32)
        scores = box_data[..., 4]
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

    def process_batch(self, batch_images, is_trained=True):
        face_tensor = torch.empty(0).to(self.device)
        eye_tensor = torch.empty(0).to(self.device)
        mouth_tensor = torch.empty(0).to(self.device)
        batch_images = batch_images.to(self.device)
        preds = self.detect(batch_images)
        bs, chl = batch_images.shape[0], batch_images.shape[1]

        for k, pred in enumerate(preds):
            img = self.draw(batch_images[k], pred)
            cv2.imshow("检测效果图", img)
            cv2.waitKey(0)
            check_set = torch.unique(pred[:, 5])
            if len(pred) == 0 or len(check_set) != 3:
                if is_trained:
                    img = self.draw(batch_images[k], pred)
                    cv2.imshow("检测效果图", img)
                    cv2.waitKey(0)
                    print(f'[ERROR]Detect: Pred{len(pred)}, Object:{len(check_set)}')
                    exit(0)
                    # TODO: 提醒操作者脸部存在遮挡
                else:
                    print('[Detect] Jump Invalid Frames')
                    face_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    eye_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    mouth_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    return face_tensor, eye_tensor, mouth_tensor

            face_score = 0
            eye_score = 0
            mouth_score = 0
            face_image = torch.empty(0).to(self.device)
            eye_image = torch.empty(0).to(self.device)
            mouth_image = torch.empty(0).to(self.device)
            for i in range(len(pred)):
                x1, y1, x2, y2, score, cls = pred[i]
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2))
                if cls == 0:
                    if score > face_score:
                        face_score = score
                        face_image = batch_images[k, :, y1:y2, x1:x2]
                        face_image = TF.resize(face_image, [224, 224]).unsqueeze(0)
                elif cls == 1:
                    if score > eye_score:
                        eye_score = score
                        eye_image = batch_images[k, :, y1:y2, x1:x2]
                        eye_image = TF.resize(eye_image, [224, 224]).unsqueeze(0)
                else:
                    if score > mouth_score:
                        mouth_score = score
                        mouth_image = batch_images[k, :, y1:y2, x1:x2]
                        mouth_image = TF.resize(mouth_image, [224, 224]).unsqueeze(0)

            face_tensor = torch.cat((face_tensor, face_image))
            eye_tensor = torch.cat((eye_tensor, eye_image))
            mouth_tensor = torch.cat((mouth_tensor, mouth_image))

        return face_tensor, eye_tensor, mouth_tensor


if __name__ == '__main__':
    device = torch.device('cuda')
    CLASSES = ['face', 'eye', 'mouth']
    image_path = "./images/test.jpg"

    image = images_numpy_to_tensor(image_path)
    image = torch.from_numpy(image)

    model = YOLODetector("./weights/yolov5n_face.pt", classes=CLASSES, o_device=device)
    face, eye, mouth = model.process_batch(image)
    print(face.shape, eye.shape, mouth.shape)
