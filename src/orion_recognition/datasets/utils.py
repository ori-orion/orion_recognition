import torch


def get_bbox_range(mean_intensity: torch.tensor, bbox_brightness_margin: float):
    assert 0 <= bbox_brightness_margin <= 1
    is_white = list((mean_intensity > 1 - bbox_brightness_margin + 1e-16).cpu().numpy())
    while True:
        try:
            min_ind = is_white.index(False)
            max_ind = len(is_white) - list(reversed(is_white)).index(False)
            break
        except ValueError:
            bbox_brightness_margin = bbox_brightness_margin * 0.9
    return min_ind, max_ind


def get_bbox(image: torch.tensor, bbox_brightness_margin):
    gray = image.mean(dim=0)
    x_min, x_max = get_bbox_range(gray.min(dim=1)[0], bbox_brightness_margin)
    y_min, y_max = get_bbox_range(gray.min(dim=0)[0], bbox_brightness_margin)
    return x_min, y_min, x_max, y_max
