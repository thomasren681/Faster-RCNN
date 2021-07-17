# ---------------------------------------------------------------------------
# This file is written by Thomas Ren at 2021/7/10
# The function implemented in this file is to do a regression for the bounding box
# It can be revealed from our test example that this regression is very close to the
# ground truth after only one step
# ---------------------------------------------------------------------------


import numpy as np
import torch

def bbox_transform(pred_bbox, gt_bbox):
    '''
    inputs:
    pred_bbox: the predicted bbox that requires to be regressed
    gt_bbox: the ground truth bbox
    both has size() (R,4) where R is the number of anchors(i.e. proposed bounding boxes)
    [Xmin,Ymin,Xmax,Ymax]

    outputs:
    gt_hat_bbox: the gt_hat_bbox which is the pred_bbox that has gone through a shift
    has size() (R,4)

    pseudo code:
    1. compute the predicted_bbox's width, height and center coordinate using the input(pred_bbox)
    2. using the ground truth and pred_bbox to compute the offset(dx,dy,dh,dw)
    3. apply the offset to the pred_bbox according to the formula
    4. get the new coordinate for the gt_hat_bbox
    (w.r.t from [ctr_x, ctr_y, height, width]-->[Xmin,Ymin,Xmax,Ymax])
    '''


    # 1. compute the predicted_bbox and gt_bbox's width, height and center coordinate using the input(pred_bbox)
    pred_widths = pred_bbox[:,2]-pred_bbox[:,0]+1 #Xmax - Xmin
    pred_heights = pred_bbox[:,3]-pred_bbox[:,1]+1 #Ymax - Ymin
    pred_ctr_x = pred_bbox[:,0]+pred_widths*0.5 #ctr_x = Xmin + 0.5*width
    pred_ctr_y = pred_bbox[:,1]+pred_heights*0.5 #ctr_y = Ymin + 0.5*height

    gt_widths = gt_bbox[:, 2] - gt_bbox[:, 0] + 1  # Xmax - Xmin
    gt_heights = gt_bbox[:, 3] - gt_bbox[:, 1] + 1  # Ymax - Ymin
    gt_ctr_x = gt_bbox[:,0] + gt_widths * 0.5  # ctr_x = Xmin + 0.5*width
    gt_ctr_y = gt_bbox[:,1] + gt_heights * 0.5

    # 2. using the ground truth and pred_bbox to compute the offset(dx,dy,dh,dw)
    dx = (gt_ctr_x-pred_ctr_x)/pred_widths
    dy = (gt_ctr_y-pred_ctr_y)/pred_heights
    dh = torch.log(gt_heights/pred_heights)
    dw = torch.log(gt_widths/pred_widths)

    # 3. apply the offset to the pred_bbox according to the formula
    gt_hat_ctr_x = dx*pred_widths+pred_ctr_x
    gt_hat_ctr_y = dy*pred_heights+pred_ctr_y
    gt_hat_width = pred_widths*torch.exp(dw)
    gt_hat_height = pred_heights*torch.exp(dh)

    # 4. get the new coordinate for the gt_hat_bbox
    # (w.r.t from [ctr_x, ctr_y, height, width]-->[Xmin,Ymin,Xmax,Ymax])
    gt_hat_bbox = torch.zeros(size=pred_bbox.size(), dtype=pred_bbox.dtype)
    gt_hat_bbox[:, 0] = gt_hat_ctr_x - 0.5*gt_hat_width # Xmin
    gt_hat_bbox[:, 2] = gt_hat_ctr_x + 0.5*gt_hat_width # Xmax
    gt_hat_bbox[:, 1] = gt_hat_ctr_y - 0.5*gt_hat_height # Ymin
    gt_hat_bbox[:, 3] = gt_hat_ctr_y + 0.5*gt_hat_height # Ymax


    return gt_hat_bbox

# if __name__ == '__main__':
#     pred_bbox = torch.tensor([[20,20,150,200]])
#     gt_bbox = torch.tensor([[25,25,175,175]])
#     gt_hat_bbox = bbox_transform(pred_bbox, gt_bbox)
#     print(gt_hat_bbox)