
import numpy as np

'''targets[(filename,uid)]=[[x1,y1,x2,y2]]
preds[t0_matched_uid]=[filename,prob,x1,y1,x2,y2,t5_matched_uids]
#class_uid_pairs = [(classname,uid)]

precision = ({relevant documents} !U {retrieved documents}) / {retrieved documents}
recall = ({relevant documents} !U {retrieved documents}) / {relevant documents}
'''
def compute_average_precision(recall, precision):
    """ Compute AP for one class.
    Args:
        recall: (numpy array) recall values of precision-recall curve.
        precision: (numpy array) precision values of precision-recall curve.
    Returns:
        (float) average precision (AP) for the class.
    """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i -1], precision[i])

    ap = 0.0 # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap

def true_positives(t0_matched_uid, t5_matched_uids):

    matched_documents = 0

    for uid in t5_matched_uids:
        if uid==t0_matched_uid:
            matched_documents+=1


    return matched_documents

def evaluate_retrieval(preds,targets,uid_names,threshold=0.5):

    aps = [] # list of average precisions (APs) for each class.
    #aps_dict = defaultdict()

    for uid_name in uid_names:
        uid_preds = preds[uid_name] # all predicted objects for this class.

        if len(uid_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---UID: {} AP {}---'.format(uid_name, ap))
            aps.append(ap)
            #aps_dict[class_name]=ap
            continue #CHECK! It should be continue as even if ap is 0 for one class, other classes would have an ap

        image_fnames = [pred[0]  for pred in uid_preds]
        probs        = [pred[1]  for pred in uid_preds]
        boxes        = [pred[2:6] for pred in uid_preds]
        t5_matched_uids = [pred[6] for pred in uid_preds]

        '''# Compute total number of ground-truth boxes for GIVEN CLASS. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, uid_name_gt) in targets:
            if uid_name_gt == uid_name:
                num_gt_boxes += len(targets[filename_gt, uid_name_gt])'''

        num_detections = len(boxes)
        n_total_relevant_documents = np.repeat(1, len(boxes))
        n_retrieved_documents = np.repeat(5,len(boxes))

        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box, t5_matched_uid) in enumerate(zip(image_fnames, boxes, t5_matched_uids)):

            if (filename, uid_name) in targets:
                boxes_gt = targets[(filename, uid_name)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if (iou >= threshold):
                        tp_uid = true_positives(uid_name,t5_matched_uid)
                        tp[det_idx] = tp_uid
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, uid_name)] # remove empty element from the dictionary.

                        break

            else:
                pass # this detection is FP.

        #tp_cumsum = np.cumsum(tp)
        #fp_cumsum = np.cumsum(fp)
        
        #precision = tp/n_retrieved_documents
        eps = np.finfo(np.float64).eps
        #precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        #recall = tp/n_relevant_documents

        precision = tp/n_retrieved_documents
        recall = tp/n_total_relevant_documents

        #ap = compute_average_precision(recall, precision)
        print('---UID {} AP {}---'.format(uid_name, precision,recall))
        # aps.append(ap)

    # Compute mAP by averaging APs for all classes.
    #print(aps)
    #from numpy import linalg as LA
    #aps = np.array(aps)
    #norm_l2 = LA.norm(aps, ord='2')
    #aps = (aps - np.min(aps))/np.ptp(aps) 
    #aps=aps/norm_l2
    print('---mAP {}---'.format(np.mean(aps)))
    return aps

def cal_ap(t0_matched_uid, t5_matched_uids,gtp=1):

    ap_uid = []
    correct_sample = 0.0

    for i,uid in enumerate(t5_matched_uids):
        if t0_matched_uid==uid:
            correct_sample+=1.0
            ap_uid.append(correct_sample/(i+1))
            print('{}/{}'.format(correct_sample,i+1))
        else:
            ap_uid.append(0.0/(i+1))
            print('{}/{}'.format(0.0,i+1))


    ap_sum = np.sum(ap_uid)
    ap = ap_sum/gtp

    return ap
    

def evaluate_retrieval_new(preds,targets,uid_names,threshold=0.5):

    aps = [] # list of average precisions (APs) for each class.
    #aps_dict = defaultdict()

    for uid_name in uid_names:
        uid_preds = preds[uid_name] # all predicted objects for this class.

        if len(uid_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---UID: {} AP {}---'.format(uid_name, ap))
            aps.append(ap)
            #aps_dict[class_name]=ap
            continue #CHECK! It should be continue as even if ap is 0 for one class, other classes would have an ap

        image_fnames = [pred[0]  for pred in uid_preds]
        probs        = [pred[1]  for pred in uid_preds]
        boxes        = [pred[2:6] for pred in uid_preds]
        t5_matched_uids = [pred[6] for pred in uid_preds]

        '''# Compute total number of ground-truth boxes for GIVEN CLASS. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, uid_name_gt) in targets:
            if uid_name_gt == uid_name:
                num_gt_boxes += len(targets[filename_gt, uid_name_gt])'''

        num_detections = len(boxes)
        n_total_relevant_documents = np.repeat(1, len(boxes))
        n_retrieved_documents = np.repeat(5,len(boxes))

        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box, t5_matched_uid) in enumerate(zip(image_fnames, boxes, t5_matched_uids)):

            if (filename, uid_name) in targets:
                boxes_gt = targets[(filename, uid_name)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if (iou >= threshold):
                        tp_uid = true_positives(uid_name,t5_matched_uid)
                        tp[det_idx] = tp_uid
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, uid_name)] # remove empty element from the dictionary.

                        break

            else:
                pass # this detection is FP.

        #tp_cumsum = np.cumsum(tp)
        #fp_cumsum = np.cumsum(fp)
        
        #precision = tp/n_retrieved_documents
        eps = np.finfo(np.float64).eps
        #precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        #recall = tp/n_relevant_documents

        precision = tp/n_retrieved_documents
        recall = tp/n_total_relevant_documents

        #ap = compute_average_precision(recall, precision)
        print('---UID {} AP {}---'.format(uid_name, precision,recall))
        sys.exit(0)
        aps.append(ap)

    # Compute mAP by averaging APs for all classes.
    #print(aps)
    #from numpy import linalg as LA
    #aps = np.array(aps)
    #norm_l2 = LA.norm(aps, ord='2')
    #aps = (aps - np.min(aps))/np.ptp(aps) 
    #aps=aps/norm_l2
    print('---mAP {}---'.format(np.mean(aps)))
    return aps


    


def evaluate(preds,targets,uid_names,threshold=0.5): #original threshold 0.5
    
    """ Compute mAP metric.
    Args:
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
        class_names: (list) list of class names.
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    my args:
        targets_ev[(filename,classname[b])].append([x1,y1,x2,y2])
        preds_ev[classname_p].append([filename, prob, x1, y1, x2, y2])

    """    
    aps = [] # list of average precisions (APs) for each class.
    #aps_dict = defaultdict()

    for uid_name in uid_names:
        uid_preds = preds[uid_name] # all predicted objects for this class.

        if len(uid_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            print('---UID: {} AP {}---'.format(class_name, ap))
            aps.append(ap)
            #aps_dict[class_name]=ap
            continue #CHECK! It should be continue as even if ap is 0 for one class, other classes would have an ap

        image_fnames = [pred[0]  for pred in uid_preds]
        probs        = [pred[1]  for pred in uid_preds]
        boxes        = [pred[2:6] for pred in uid_preds]
        t5_matched_uids = [pred[6] for pred in uid_preds]

        '''sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes        = [boxes[i]        for i in sorted_idxs]
        t5_matched_uids = [t5_matched_uids[i] for i in sorted_idxs]'''

        # Compute total number of ground-truth boxes for GIVEN CLASS. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, uid_name_gt) in targets:
            if uid_name_gt == uid_name:
                num_gt_boxes += len(targets[filename_gt, uid_name_gt])
                #num_gt_boxes +=2

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box, t5_matched_uid) in enumerate(zip(image_fnames, boxes, t5_matched_uids)):

            if (filename, uid_name) in targets:
                boxes_gt = targets[(filename, uid)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if (iou >= threshold) and (uid_name==uid) and (uid in t5_matched_uid):
                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, uid_name)] # remove empty element from the dictionary.

                        break

            else:
                pass # this detection is FP.
        
        # Compute AP from `tp` and `fp`.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        ap = compute_average_precision(recall, precision)
        print('---class {} AP {}---'.format(class_name, ap))
        aps.append(ap)
        #aps_dict[class_name]=ap

    # Compute mAP by averaging APs for all classes.
    print('---mAP {}---'.format(np.mean(aps)))
    return aps


