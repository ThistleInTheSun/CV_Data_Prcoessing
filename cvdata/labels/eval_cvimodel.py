import argparse
import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm


def load_txt(txt_path, empty_shape=(0, 5)):
    bboxes = []
    if osp.exists(txt_path):
        with open(txt_path) as f:
            for line in f.readlines():
                line = line.strip().split()
                line = [float(_) for _ in line]
                bboxes.append(line)
    if bboxes:
        return torch.tensor(bboxes)
    else:
        return torch.empty(empty_shape)


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
    
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def main(image_dir, glabel_dir, plabel_dir, gformat, pformat):
    def process_one(detections, labels):
        if gformat == 'xywh':
            labels[:, 1:3] -= labels[:, 3:5] / 2
            labels[:, 3:5] += labels[:, 1:3]

        if pformat == 'xywh':
            detections[:, 1:3] -= detections[:, 3:5] / 2
            detections[:, 3:5] += detections[:, 1:3]

        iou = box_iou(labels[:, 1:], detections[:, 1:5])
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 0]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)
    
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    stats = []
    images = sorted(os.listdir(image_dir))
    labels = [_.rsplit('.', 1)[0] + '.txt' for _ in images]
    seen = 0

    for label in tqdm(labels):
        seen += 1
        pred = load_txt(osp.join(plabel_dir, label), empty_shape=(0, 6))
        pred = pred[pred[:, -1].argsort(descending=True)]
        gt = load_txt(osp.join(glabel_dir, label))
        npr = len(pred)
        nl = len(gt)
        cls = gt[:, 0]
        correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool)
        if npr == 0:
            if nl:
                stats.append((correct_bboxes, *torch.zeros((2, 0)), cls))
            continue
        if nl:
            correct_bboxes = process_one(pred.clone(), gt.clone())

        stats.append((correct_bboxes, pred[:, 5], pred[:, 0], cls))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    result = ap_per_class(*stats)[2:]
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help='the dir path of images')
    parser.add_argument('--glabels', help='the dir path of ground truth label txts')
    parser.add_argument('--plabels', help='the dir path of prediction label txts')
    parser.add_argument('--gformat', default='xyxy')
    parser.add_argument('--pformat', default='xyxy')
    args = parser.parse_args()

    if args.glabels is None: args.glabels = args.images.replace('images', 'labels')
    result = main(args.images, args.glabels, args.plabels, args.gformat, args.pformat)
    print('{:>10s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('class', 'p', 'r', 'map50', 'map'))
    print('{:>10s} {:10g} {:10g} {:10g} {:10g}'.format('all', 
                                                   result[0].mean(),
                                                   result[1].mean(),
                                                   result[3][:, 0].mean(),
                                                   result[3].mean()))
    for i in range(len(result[0])):
        print('{:10g} {:10g} {:10g} {:10g} {:10g}'.format(i, 
                                                    result[0][i],
                                                    result[1][i],
                                                    result[3][i, 0],
                                                    result[3][i].mean()))
    print()