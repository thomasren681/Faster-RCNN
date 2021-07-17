

from bbox import bbox_overlaps
import numpy as np
import torch


def compute_IoU(pred_bbox,gt_bbox):
    '''
    inputs:
    pred_bbox: the predicted bbox that requires to be regressed
    has size() [R,4]
    gt_bbox: the ground truth bbox
    has size() [N,4]
    each row contains [Xmin,Ymin,Xmax,Ymax]

    outputs:
    IoU: has size(R,N), is also a symmetric tensor
    each row contains the IoU of a pred_bbox to every gt_bbox

    pseudo code:
    1. compute the area of pred_bbox and gt_bbox (Union)
    2. compute the area of the intersection region (Intersection)
    3. divided aforementioned two area to get the final IoU
    '''

    # 1. compute the area of pred_bbox and gt_bbox (Union)
    pred_widths = pred_bbox[:, 2] - pred_bbox[:, 0] + 1  # Xmax - Xmin
    pred_heights = pred_bbox[:, 3] - pred_bbox[:, 1] + 1  # Ymax - Ymin
    pred_area = pred_heights*pred_widths

    gt_widths = gt_bbox[:, 2] - gt_bbox[:, 0] + 1  # Xmax - Xmin
    gt_heights = gt_bbox[:, 3] - gt_bbox[:, 1] + 1  # Ymax - Ymin
    gt_area = gt_heights*gt_widths

    # 2. compute the area of the intersection region (Intersection)
    # the trick of computing the intersection region's four coordinates is that
    # Xmin = the maximum of the Xmin; Xmax = the minimum of the Xmax
    # Ymin = the maximum of the Ymin; Ymax = the minimum of the Ymax
    # Notice the use of broadcast here: using two (R,) tensor to get a (R,R) tensor
    intersection_widths = (1+(torch.min(pred_bbox[:,2].view(-1,1), gt_bbox[:,2].t().view(1,-1)))-\
                          (torch.max(pred_bbox[:,0].view(-1,1), gt_bbox[:,0].t().view(1,-1)))).clamp(min=0)
    intersection_heights = (1+(torch.min(pred_bbox[:, 3].view(-1, 1), gt_bbox[:, 3].t().view(1, -1))) - \
                          (torch.max(pred_bbox[:, 1].view(-1, 1), gt_bbox[:, 1].t().view(1, -1)))).clamp(min=0)
    intersection_area = intersection_heights*intersection_widths
    # .clamp(min = 0) is equal to ReLU function but has a derivative of 1 when x is at zero
    # while ReLU has a derivative of 0 when x is at zero
    # This is to assure that when there is no intersection between, the intersection area should be zero
    # while according to the original algorithm(without clamp) would output a negative value

    # 3. divided aforementioned two area to get the final IoU
    union_area = pred_area.view(-1,1) + gt_area.view(1,-1) - intersection_area
    IoU = intersection_area/union_area

    return IoU


if __name__ == '__main__':
    boxes = torch.randn(3, 4)
    query_boxes = torch.rand(5, 4)
    c = compute_IoU(boxes, query_boxes)
    d = bbox_overlaps(boxes, query_boxes)
    print('The difference for our code and the original code is ', np.linalg.norm(c-d))
    print('the difference should be less than 1e-07 for any test case')