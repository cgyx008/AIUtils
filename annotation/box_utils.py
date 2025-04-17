import numpy as np


def xyxy2xcycwh(xyxy, w=1, h=1):
    """Convert [xmin, ymin, xmax, ymax] to [xc, yc, w, h] with w/h normalization

    Args:
        xyxy (Sequence): [[xmin, ymin, xmax, ymax]] or [xmin, ymin, xmax, ymax]
        w (int): image width
        h (int): image height

    Return:
        (np.ndarray): normalized [xc, yc, w, h]
    """
    boxes = np.array(xyxy, dtype=float)
    ndim = boxes.ndim

    boxes = boxes.reshape(-1, 4)
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h
    boxes = boxes.clip(min=0, max=1)
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    boxes[:, 0] += boxes[:, 2] / 2
    boxes[:, 1] += boxes[:, 3] / 2
    return boxes if ndim == 2 else boxes[0]
