import torch
from torch.autograd import Variable
import torch.nn as nn

from utils_2016310526 import *
import torchvision.transforms as transforms
import cv2
import numpy as np


def decoder(pred):
    grid_num = 14
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  # 7x7x30
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.stack((contain1, contain2), 2)
    mask1 = contain > 0.1
    mask2 = (contain == contain.max())
    mask = (mask1 + mask2).gt(0)

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == True:
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4)[0])
                        cls_indexs.append(cls_index)
                        probs.append((contain_prob * max_prob)[0])
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.stack(boxes, 0)
        probs = torch.stack(probs, 0)
        cls_indexs = torch.stack(cls_indexs, 0)
    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.5):

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            break

        if order.dim() == 0:
            continue

        i = order[0]
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero(as_tuple=False).squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


#
# start predict one image
#
def predict_gpu(model, image_name, root_path, device):
    result = []
    image = cv2.imread(root_path + image_name)
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)

    pred = model(img)  # 1x7x7x30
    pred = pred.cpu()
    boxes, cls_indexs, probs = decoder(pred)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])

    return result


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root_dataset_dirpath = './VOCdevkit/VOC2012/'

    from tqdm import tqdm
    from collections import defaultdict

    target = defaultdict(list)
    preds = defaultdict(list)
    test_img = []

    f = open('./2016310526_VOC_TEST_DATA.txt')

    lines = f.readlines()
    file_list = []

    for line in lines:
        splitted = line.strip().split()
        file_list.append(splitted)
    f.close()

    print('----------read txt file done----------')

    for i, image_file in enumerate(file_list):
        image_id = image_file[0]

        test_img.append(image_id)
        num_boxes = (len(image_file) - 1) // 5
        # print(test_img)
        # print(num_boxes)

        for i in range(num_boxes):
            xmin = int(image_file[1+5*i])
            ymin = int(image_file[2+5*i])
            xmax = int(image_file[3+5*i])
            ymax = int(image_file[4+5*i])
            class_idx = int(image_file[5+5*i])
            class_name = VOC_CLASSES[class_idx]
            target[(image_id, class_name)].append([xmin, ymin, xmax, ymax])

    print('------------parsing finished------------')

    print('---------------start test---------------')
    model = resnet50()
    print('load model...model_2016310526.pth...')
    model.load_state_dict(torch.load('model_2016310526.pth'))
    model.eval()
    model.to(device)

    import os
    for img_path in tqdm(test_img):
        result = predict_gpu(model, img_path, root_path=os.path.join(root_dataset_dirpath, 'JPEGImages/'), device=device)
        image = cv2.imread(os.path.join(root_dataset_dirpath, 'JPEGImages/') + img_path)
        for left_up, right_bottom, class_name, _, prob in result:
            color = Color[VOC_CLASSES.index(class_name)]
            cv2.rectangle(image, left_up, right_bottom, color, 2)
            label = class_name + str(round(prob, 2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
        if not os.path.exists('./test_images'):
            os.mkdir('./test_images')

        cv2.imwrite('./test_images/{}_result.jpg'.format(img_path), image)