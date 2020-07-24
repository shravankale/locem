import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

import os
import cv2
import numpy as np
import pandas as pd

from darknet import DarkNet
from yolo_v1 import YOLOv1
#from main.nn_view import View


'''def visualize_boxes(image_bgr, boxes, class_names, probs, name_bgr_dict=None, line_thickness=2):
    if name_bgr_dict is None:
        name_bgr_dict = VOC_CLASS_BGR

    image_boxes = image_bgr.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        # Draw box on the image.
        left_top, right_bottom = box
        left, top = int(left_top[0]), int(left_top[1])
        right, bottom = int(right_bottom[0]), int(right_bottom[1])
        bgr = name_bgr_dict[class_name]
        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        # Draw text on the image.
        text = '%s %.2f' % (class_name, prob)
        size, baseline = cv2.getTextSize(text,  cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        text_w, text_h = size

        x, y = left, top
        x1y1 = (x, y)
        x2y2 = (x + text_w + line_thickness, y + text_h + line_thickness + baseline)
        cv2.rectangle(image_boxes, x1y1, x2y2, bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2*baseline + line_thickness),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)

    return image_boxes'''


class yoloDetector:
    def __init__(self,
        model, class_name_list=None, mean_rgb=[122.67891434, 116.66876762, 104.00698793],
        conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.5,
        gpu_id=2,S=7,B=2,C=30,X=5,beta=64,image_size=448):

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        use_gpu = torch.cuda.is_available()
        assert use_gpu, 'Current implementation does not support CPU mode. Enable CUDA.'

        map_vid = pd.read_pickle("../data/map_vid.pkl")
        self.class_name_list = list(map_vid['category_name'])

        self.S, self.B, self.C, self.beta = S,B,C,beta


        '''self.mean = np.array(mean_rgb, dtype=np.float32)
        assert self.mean.shape == (3,)'''

        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.gpu_id = gpu_id
        self.image_size=image_size

        self.to_tensor = transforms.ToTensor()
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        #Fetch locEm model
        self.yolo_imgnetvid = model
        self.yolo_imgnetvid.eval()
    
    '''def getModel(self,model_path):

        S,B,C=self.S, self.B, self.C
        X=5

        print("Loading model from ",model_path)
        model = models.__dict__['resnet50']()
        num_ftrs = model.fc.in_features

        num_classes = S*S*(B*X+C)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs,4096),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(),
            #nn.Dropout(0.5, inplace=False),
            nn.Linear(4096,num_classes),
            nn.Sigmoid(),
            View((-1,S,S,B*X+C))
        )

        model = torch.nn.DataParallel(model).cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
        loc = 'cuda:{}'.format(self.gpu_id)
        checkpoint = torch.load(model_path, map_location=loc)


        #best_acc1 = checkpoint['best_acc1']
        #epch = checkpoint['epoch']
        #print("Best Training Accuracy: {} @Epoch: {}".format(best_acc1,epch))

        model.load_state_dict(checkpoint)
        #model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])

        return model'''

    def detect(self, img):
        """ Detect objects from given image.
        Args:
            image_bgr: (numpy array) input image in BGR ids_sorted, sized [h, w, 3].
            image_size: (int) image width and height to which input image is resized.
        Returns:
            boxes_detected: (list of tuple) box corner list like [((x1, y1), (x2, y2))_obj1, ...]. Re-scaled for original input image size.
            class_names_detected: (list of str) list of class name for each detected boxe.
            probs_detected: (list of float) list of probability(=confidence x class_score) for each detected box.
        """

        S,B,C = self.S, self.B, self.C

        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # assuming the model is trained with RGB images.
        img = (img - self.mean) / 255.0
        img = self.to_tensor(img) # [image_size, image_size, 3] -> [3, image_size, image_size]
        img = img[None, :, :, :]  # [3, image_size, image_size] -> [1, 3, image_size, image_size]
        img = Variable(img)
        img = img.cuda()
        #img = img.to(torch.device('cuda:1'))

        with torch.no_grad():
            pred_tensor, pred_embedding = self.yolo_imgnetvid(img)
        pred_embedding = pred_embedding.clone().detach()
        pred_tensor = pred_tensor.cpu().data
        pred_tensor_output = pred_tensor.clone().detach()
        pred_tensor = pred_tensor.squeeze(0) # squeeze batch dimension.

        #print('pred_tensor from detect',pred_tensor_output.size())

        # Get detected boxes_detected, labels, confidences, class-scores.
        boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = self.decode(pred_tensor)
        if boxes_normalized_all.size(0) == 0:
            return [], [], [],[] # if no box found, return empty lists.

        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs = [], [], []

        for class_label in range(len(self.class_name_list)):
            mask = (class_labels_all == class_label)
            if torch.sum(mask) == 0:
                continue # if no box found, skip that class.

            boxes_normalized_masked = boxes_normalized_all[mask]
            class_labels_maked = class_labels_all[mask]
            confidences_masked = confidences_all[mask]
            class_scores_masked = class_scores_all[mask]

            ids = self.nms(boxes_normalized_masked, confidences_masked)

            boxes_normalized.append(boxes_normalized_masked[ids])
            class_labels.append(class_labels_maked[ids])
            probs.append(confidences_masked[ids] * class_scores_masked[ids])

        boxes_normalized = torch.cat(boxes_normalized, 0)
        class_labels = torch.cat(class_labels, 0)
        probs = torch.cat(probs, 0)

        # Postprocess for box, labels, probs.
        boxes_detected, class_names_detected, probs_detected = [], [], []
        for b in range(boxes_normalized.size(0)):
            box_normalized = boxes_normalized[b]
            class_label = class_labels[b]
            prob = probs[b]

            x1, x2 = w * box_normalized[0], w * box_normalized[2] # unnormalize x with image width.
            y1, y2 = h * box_normalized[1], h * box_normalized[3] # unnormalize y with image height.
            boxes_detected.append(((x1, y1), (x2, y2)))

            class_label = int(class_label) # convert from LongTensor to int.
            class_name = self.class_name_list[class_label]
            class_names_detected.append(class_name)

            prob = float(prob) # convert from Tensor to float.
            probs_detected.append(prob)

        return boxes_detected, class_names_detected, probs_detected,pred_tensor_output

    def decode(self, pred_tensor):
        """ Decode tensor into box coordinates, class labels, and probs_detected.
        Args:
            pred_tensor: (tensor) tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        Returns:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
            labels: (tensor) class labels for each detected boxe, sized [n_boxes,].
            confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].
            class_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].
        """
        S, B, C = self.S, self.B, self.C
        boxes, labels, confidences, class_scores = [], [], [], []

        cell_size = 1.0 / float(S)

        conf = pred_tensor[:, :, 4].unsqueeze(2) # [S, S, 1]
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5*b + 4].unsqueeze(2)), 2)
        conf_mask = conf > self.conf_thresh # [S, S, B]

        # TBM, further optimization may be possible by replacing the following for-loops with tensor operations.
        for i in range(S): # for x-dimension.
            for j in range(S): # for y-dimension.
                class_score, class_label = torch.max(pred_tensor[j, i, 5*B:], 0)

                for b in range(B):
                    conf = pred_tensor[j, i, 5*b + 4]
                    prob = conf * class_score
                    if float(prob) < self.prob_thresh:
                        continue

                    # Compute box corner (x1, y1, x2, y2) from tensor.
                    box = pred_tensor[j, i, 5*b : 5*b + 4]
                    x0y0_normalized = torch.FloatTensor([i, j]) * cell_size # cell left-top corner. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    xy_normalized = box[:2] * cell_size + x0y0_normalized   # box center. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    wh_normalized = box[2:] # Box width and height. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    box_xyxy = torch.FloatTensor(4) # [4,]
                    box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized # left-top corner (x1, y1).
                    box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized # right-bottom corner (x2, y2).

                    # Append result to the lists.
                    boxes.append(box_xyxy)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_scores.append(class_score)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0) # [n_boxes, 4]
            labels = torch.stack(labels, 0)             # [n_boxes, ]
            confidences = torch.stack(confidences, 0)   # [n_boxes, ]
            class_scores = torch.stack(class_scores, 0) # [n_boxes, ]
        else:
            # If no box found, return empty tensors.
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)
            confidences = torch.FloatTensor(0)
            class_scores = torch.FloatTensor(0)

        return boxes, labels, confidences, class_scores

    def nms(self, boxes, scores):
        """ Apply non maximum supression.
        Args:
        Returns:
        """
        threshold = self.nms_thresh

        x1 = boxes[:, 0] # [n,]
        y1 = boxes[:, 1] # [n,]
        x2 = boxes[:, 2] # [n,]
        y2 = boxes[:, 3] # [n,]
        areas = (x2 - x1) * (y2 - y1) # [n,]

        _, ids_sorted = scores.sort(0, descending=True) # [n,]
        ids = []
        while ids_sorted.numel() > 0:
            # Assume `ids_sorted` size is [m,] in the beginning of this iter.

            i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
            ids.append(i)

            if ids_sorted.numel() == 1:
                break # If only one box is left (i.e., no box to supress), break.

            inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i]) # [m-1, ]
            inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i]) # [m-1, ]
            inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i]) # [m-1, ]
            inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i]) # [m-1, ]
            inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

            inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
            unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
            ious = inters / unions # [m-1, ]

            # Remove boxes whose IoU is higher than the threshold.
            ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
            if ids_keep.numel() == 0:
                break # If no box left, break.
            ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

        return torch.LongTensor(ids)


'''if __name__ == '__main__':
    # Paths to input/output images.
    image_path = 'data/test_samples/009046.jpg'
    out_path = 'result.png'
    # Path to the yolo weight.
    model_path = 'weights/yolo/model_best.pth'
    # GPU device on which yolo is loaded.
    gpu_id = 0

    # Load model.
    yolo = YOLODetector(model_path, gpu_id=gpu_id, conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.35)

    # Load image.
    image = cv2.imread(image_path)

    # Detect objects.
    boxes, class_names, probs = yolo.detect(image)

    # Visualize.
    image_boxes = visualize_boxes(image, boxes, class_names, probs)

    # Output detection result as an image.
    cv2.imwrite(out_path, image_boxes)'''
