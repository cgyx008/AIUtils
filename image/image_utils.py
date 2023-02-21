import cv2


def draw_rect_and_put_text(img, box, text, color=(0, 0, 255), box_thickness=1,
                           font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6):
    # bounding-box
    h, w, _ = img.shape
    box = list(map(int, box))
    pt1 = (max(box[0], 0), max(box[1], 0))
    pt2 = (min(box[2], w - 1), min(box[3], h - 1))
    cv2.rectangle(img, pt1, pt2, color, box_thickness)

    # text-box
    tbox_color = color
    text_size = cv2.getTextSize(text, font, font_scale, box_thickness * 2)
    if pt1[1] - text_size[0][1] < 0:
        tbox_pt1 = (pt1[0], pt1[1])
        tbox_pt2 = (pt1[0] + text_size[0][0], pt1[1] + text_size[0][1])
    else:
        tbox_pt1 = (pt1[0], pt1[1] - text_size[0][1])
        tbox_pt2 = (pt1[0] + text_size[0][0], pt1[1])
    cv2.rectangle(img, tbox_pt1, tbox_pt2, tbox_color, -1)

    # text
    if pt1[1] - text_size[0][1] < 0:
        text_pt = (pt1[0], pt1[1] + text_size[0][1])
    else:
        text_pt = pt1
    tcolor = (255, 255, 255)
    cv2.putText(img, text, text_pt, font, font_scale, tcolor, box_thickness * 2)
    return img
