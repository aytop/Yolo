import YoloNet
import torch
import numpy as np
import Util
import cv2


class Inference:
    def __init__(self):
        self.model = YoloNet.YOLONet()
        self.model.load_state_dict(torch.load('yolo_cpu.pt'))

    def __call__(self, image):
        org_size = image.shape  # (480, 640, 3) HxWxC
        img = self.rescale(image, (144, 72))
        img = self.to_tensor(img)
        with torch.no_grad():
            output = self.model(img)
        boxes = self.parse_output(output)
        boxes = self.scale_boxes(boxes, org_size[:2])
        return self.draw(image, boxes)

    @staticmethod
    def rescale(image, size):
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def to_tensor(image):
        image = image.transpose((2, 0, 1))
        image[0, :, :], image[2, :, :] = image[2, :, :], image[0, :, :]
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        image = image.float()
        return torch.autograd.Variable(image)

    @staticmethod
    def parse_output(output):
        return Util.NMS(output, data_size=(72, 144))

    @staticmethod
    def draw(img, boxes):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for box in boxes:
            cv2.rectangle(img, box['top-left'], box['bottom-right'], color=(0, 0, 255), thickness=1)
            cv2.putText(img=img,
                        text=str(box['class']),
                        org=box['top-left'],
                        bottomLeftOrigin=False,
                        fontFace=font,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)
        return img

    @staticmethod
    def scale_boxes(boxes, size):
        org_h, org_w = size
        ratio = org_w/144, org_h/72
        for box in boxes:
            # x je width; y je height
            print(box['class'], box['confidence'], box['top-left'])
            box['top-left'] = box['top-left'][0]*ratio[0], box['top-left'][1]*ratio[1]
            box['bottom-right'] = box['bottom-right'][0]*ratio[0], box['bottom-right'][1]*ratio[1]
        return boxes
