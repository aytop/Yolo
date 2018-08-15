import numpy as np
from torch.nn import functional as f
from PIL import Image, ImageDraw,ImageFont
import pathlib


def iou(tl1, br1, tl2, br2):
    (tx1, ty1) = tl1
    (tx2, ty2) = tl2
    (x1, y1) = (max(tx1, tx2), max(ty1, ty2))
    (bx1, by1) = br1
    (bx2, by2) = br2
    (x2, y2) = (min(bx1, bx2), min(by1, by2))
    if x1 > x2 or y1 > y2:
        return 0
    i = (x2 - x1) * (y2 - y1)
    p1 = (bx1 - tx1) * (by1 - ty1)
    p2 = (bx2 - tx2) * (by2 - ty2)
    return i/(p1+p2-i)


def get_boxes(output, data_size):
    boxes = []
    for i in range(9):
        for j in range(9):
            class_prob, (tl, br), c = output[0, :10, i, j],\
                                      findAbsoluteCoor(data_size, output, i, j), output[0, 14, i, j]
            class_prob = f.softmax(class_prob)
            clas = np.array(class_prob).argmax()
            if not (tl == br and c == 0):
                boxes.append({'confidence': c, 'top-left': tl, 'bottom-right': br, 'class': clas})
    return boxes


def sort_boxes(boxes):
    if len(boxes) < 2:
        return boxes
    pivot = boxes[0]
    lt = []
    gt = []
    for box in boxes[1:]:
        if box['confidence'] < pivot['confidence']:
            lt.append(box)
        else:
            gt.append(box)
    lt = sort_boxes(lt)
    gt = sort_boxes(gt)
    gt.append(pivot)
    return gt+lt


# returns non-maximal suppressed list of boxes
def NMS(output, data_size,  confidence_threshold=0.2, iou_threshold=0.4):
    conf_mask = (output[:, 14, :, :] > confidence_threshold).float()
    output = output * conf_mask  # delete boxes with low confidence
    boxes = get_boxes(output, data_size)
    boxes = sort_boxes(boxes)
    boxes = [[box, True] for box in boxes]
    ret = []
    for index, box in enumerate(boxes):
        if box[1]:
            ret.append(box[0])
            for i in range(index+1, len(boxes)):
                 if (iou(boxes[i][0]['top-left'], boxes[i][0]['bottom-right'], box[0]['top-left'], box[0]['bottom-right']) > iou_threshold):
                        boxes[i][1] = False
    return ret


def findAbsoluteCoor(data_size, label, i, j):
    img_w = data_size[1]
    img_h = data_size[0]
    reg_w = img_w / 9
    reg_h = img_h / 9
    cx1, cy1, w1, h1 = label[0, 10:14, i, j]
    w1, h1 = w1 * img_w, h1 * img_h
    box = ((cx1 + reg_w * i - w1 / 2, cy1 + reg_h * j - h1 / 2),
           (cx1 + reg_w * i + w1 / 2, cy1 + reg_h * j + h1 / 2))
    return box


def toImage(boxes, image, path, index):
    image = np.array(image, dtype='uint8')[0]
    image = np.transpose(image, (1, 2, 0))
    img = Image.fromarray(image, 'RGB')
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box['top-left'], box['bottom-right']], outline=(255, 0, 0))
        draw.text(box['top-left'], str(box['class']), fill=(255, 0, 0), font=ImageFont.load_default())
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    img.save(path+'/'+str(index)+'.png')


def score(image, truth, prediction, path, index, iou_threshold=0.5):
    pred_boxes = NMS(prediction, image.size()[2:4])
    truth_boxes = NMS(truth, image.size()[2:4])
    # toImage(pred_boxes, image, path+'/outputs', index)
    # toImage(truth_boxes, image, path+'/ground truth', index)
    ret = {'hit': 0, 'miss': 0, 'miss_class': 0}
    for pBox, tBox in zip(pred_boxes, truth_boxes):
        if iou(pBox['top-left'], pBox['bottom-right'], tBox['top-left'], tBox['bottom-right']) > iou_threshold:
            if pBox['class'] == tBox['class']:
                ret['hit'] += 1
            else:
                ret['miss_class'] += 1
        else:
            ret['miss'] += 1
    return ret


def anc_score(image, truth, prediction, path, index, iou_threshold=0.5):
    pred_boxes = []
    truth_boxes = []
    for anchor in range(3):
        pred_boxes += NMS(prediction[:, anchor * 15:(anchor+1) * 15, :, :], image.size()[2:4])
        truth_boxes += NMS(truth[:, anchor * 15:(anchor + 1) * 15, :, :], image.size()[2:4])
    # toImage(pred_boxes, image, path+'/outputs', index)
    # toImage(truth_boxes, image, path+'/ground truth', index)
    ret = {'hit': 0, 'miss': 0, 'miss_class': 0, 'hit_class': 0}
    for pBox, tBox in zip(pred_boxes, truth_boxes):
        if iou(pBox['top-left'], pBox['bottom-right'], tBox['top-left'], tBox['bottom-right']) > iou_threshold:
            if pBox['class'] == tBox['class']:
                ret['hit'] += 1
            else:
                ret['miss_class'] += 1
        else:
            if pBox['class'] == tBox['class']:
                ret['hit_class'] += 1
            else:
                ret['miss'] += 1
    return ret

