import torch


def cal_iou(outputs, labels, SMOOTH=1e-6):
    with torch.no_grad():
        outputs = outputs.squeeze(1).bool()  # BATCH x 1 x H x W => BATCH x H x W
        labels = labels.squeeze(1).bool()

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou


def get_iou_score(outputs, labels):
    A = labels.squeeze(1)[0].bool()
    pred = torch.where(outputs < 0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    B = pred.squeeze(1)[0].bool()
    intersection = (A & B).float().sum((1, 2))
    union = (A | B).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.cpu().detach().numpy()