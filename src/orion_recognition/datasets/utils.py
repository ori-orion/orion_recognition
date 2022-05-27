import torch


def get_bbox_range(mean_intensity: torch.tensor, bbox_brightness_thresh):
    is_white = list((mean_intensity > bbox_brightness_thresh).cpu().numpy())
    min_ind = is_white.index(False)
    max_ind = len(is_white) - list(reversed(is_white)).index(False)
    return min_ind, max_ind


def get_bbox(image: torch.tensor, bbox_brightness_thresh):
    gray = image.mean(dim=0)
    x_min, x_max = get_bbox_range(gray.min(dim=1)[0], bbox_brightness_thresh)
    y_min, y_max = get_bbox_range(gray.min(dim=0)[0], bbox_brightness_thresh)
    return x_min, y_min, x_max, y_max
