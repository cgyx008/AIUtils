import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())
    sys.path.insert(0, (Path(__file__).parents[2] / 'ultralytics').as_posix())
from ultralytics.utils.plotting import Annotator, colors

from image.image_utils import draw_rect_and_put_text, keep_wh_resize
from video.video_utils import get_cap_and_attr


def read_txt(txt_path, w=1, h=1):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = {}
    frame_idx, cls = -1, None
    for line in lines:
        match = re.match('pts', line)
        if match:
            frame_idx += 1
            results[frame_idx] = defaultdict(list)
            continue

        match = re.match('\t([a-z]+):', line)
        if match:
            cls = match.group(1)
            if (w == 1 and h == 1) or (w == 0 and h == 0):
                w = int(re.search(r'width:\s*(\d+)', line).group(1))
                h = int(re.search(r'height:\s*(\d+)', line).group(1))
            continue

        match = re.findall(r'\d+', line)
        if match:
            match = list(map(int, match))
            match[2] += match[0]
            match[3] += match[1]
            match[0] /= w
            match[1] /= h
            match[2] /= w
            match[3] /= h
            results[frame_idx][cls].append(match)

    return results


def get_model_results(video_path):
    model_txt = video_path.parent / f'{video_path.stem}-model.txt'
    return read_txt(model_txt)


def get_action_results(video_path):
    action_txt = video_path.parent / f'{video_path.stem}-action.txt'
    return read_txt(action_txt)


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_str = f.read()
    json_str = json_str.replace('}\n{', '},\n{')
    json_str = f'[{json_str}]'
    return json.loads(json_str)


def parse_video(video_path, num_workers=8, vis_w=640, vis_h=352):
    """
    Examples:
        >>> parse_video(r'H:\data/reolink\test\20251218\1\diff_peo\day\c15_fn\b26\RecM0A_20251217_161313_161812_0_531ED300000000_4D897C7.mp4')
    """
    video_path = Path(video_path)

    ai_infos = read_json(video_path.parent / f'{video_path.stem}.json')
    # model_results = get_model_results(video_path)
    # action_results = get_action_results(video_path)

    cap, width, height, num_frames, fps, *_ = get_cap_and_attr(video_path)
    num_0s = len(str(num_frames))
    save_dir = video_path.parent / video_path.stem / 'model_and_action'
    save_dir.mkdir(parents=True, exist_ok=True)
    action_dir = video_path.parent / video_path.stem / 'action'
    action_dir.mkdir(parents=True, exist_ok=True)

    # colors = {'ad': (255, 0, 0), 'pd': (0, 255, 0), 'vd': (0, 0, 255)}
    executor = ThreadPoolExecutor(num_workers) if num_workers else None
    # in_alarm_interval = 0  # 8 * fps
    # for i, (mo, ao) in enumerate(zip(tqdm(model_results), action_results)):
    #     ret, model_frame = cap.read()
    #     action_frame = model_frame.copy()
    #     action = False
    #     for cls, outputs in model_results[mo].items():
    #         color = colors[cls]
    #         for *box, score in outputs:
    #             if len(box) > 4:
    #                 score = box[4]
    #             box[0] *= width
    #             box[1] *= height
    #             box[2] *= width
    #             box[3] *= height
    #             text = f'{cls} {score}'
    #             draw_rect_and_put_text(model_frame, box, text, color, 2)
    #
    #     for cls, outputs in action_results[ao].items():
    #         color = colors[cls]
    #         for *box, score, rid, state in outputs:
    #             if state != 1:
    #                 continue
    #             if len(box) > 4:
    #                 score = box[4]
    #             action = True
    #             box[0] *= width
    #             box[1] *= height
    #             box[2] *= width
    #             box[3] *= height
    #             text = f'{cls} {score}'
    #             draw_rect_and_put_text(action_frame, box, text, color, 2)
    #
    #     img = cv2.hconcat([model_frame, action_frame])
    #     save_name = f'{video_path.stem}_{str(i).zfill(num_0s)}.jpg'
    #     save_path = save_dir / save_name
    #     if executor is not None:
    #         executor.submit(cv2.imwrite, str(save_path), img)  # noqa
    #     else:
    #         cv2.imwrite(save_path.as_posix(), img)
    #     if action:
    #         save_path = action_dir / save_name
    #         if executor is not None:
    #             executor.submit(cv2.imwrite, str(save_path), img)  # noqa
    #         else:
    #             cv2.imwrite(save_path.as_posix(), img)
    for i, ai_info in enumerate(tqdm(ai_infos)):
        ret, model_frame = cap.read()
        if not ret:
            continue
        # if 0 < in_alarm_interval < 5 * fps:
        #     in_alarm_interval += 1
        #     continue
        # in_alarm_interval = 0

        model_frame = keep_wh_resize(model_frame, (vis_w, vis_h))
        height, width = model_frame.shape[:2]
        action_frame = model_frame.copy()

        annotator = Annotator(model_frame)
        for j, (cat, cat_dict) in enumerate(ai_info['model'].items()):
            for obj in cat_dict['objs']:
                ai_stream_w, ai_stream_h = cat_dict['width'], cat_dict['height']
                box = [
                    obj['x'] / ai_stream_w * width,
                    obj['y'] / ai_stream_h * height,
                    (obj['x'] + obj['w']) / ai_stream_w * width,
                    (obj['y'] + obj['h']) / ai_stream_h * height,
                ]
                text = f'{cat} {obj["score"]}'
                annotator.box_label(box, text, colors(j)[::-1])
        model_frame = annotator.result()

        annotator = Annotator(action_frame)
        action = False
        for j, (cat, cat_dict) in enumerate(ai_info['action'].items()):
            # if cat != 'pd':
            #     continue
            for obj in cat_dict['objs']:
                if obj['state'] != 1:
                    continue
                action = True
                ai_stream_w, ai_stream_h = cat_dict['width'], cat_dict['height']
                box = [
                    obj['x'] / ai_stream_w * width,
                    obj['y'] / ai_stream_h * height,
                    (obj['x'] + obj['w']) / ai_stream_w * width,
                    (obj['y'] + obj['h']) / ai_stream_h * height,
                ]
                text = f'{cat} {obj["score"]}'
                annotator.box_label(box, text, colors(j)[::-1])
        action_frame = annotator.result()

        # in_alarm_interval = 1 if action else in_alarm_interval + 1

        img = cv2.hconcat([model_frame, action_frame])
        # img = action_frame
        save_name = f'{video_path.stem}_{str(i).zfill(num_0s)}.jpg'
        save_path = save_dir / save_name
        # if executor is not None:
        #     executor.submit(cv2.imwrite, str(save_path), img)  # noqa
        # else:
        #     cv2.imwrite(save_path.as_posix(), img)
        if action:
            save_path = action_dir / save_name
            if executor is not None:
                executor.submit(cv2.imwrite, str(save_path), action_frame)  # noqa
            else:
                cv2.imwrite(save_path.as_posix(), action_frame)


def parse_videos(video_dir, num_workers=8):
    """
    Examples:
        >>> parse_videos(r'H:\data/reolink\test\20251218\1\diff_peo\day\c15_fn\c15', 8)
    """
    video_paths = sorted(Path(video_dir).glob('*.mp4'))
    for i, video_path in enumerate(video_paths):
        print(f'{i + 1} / {len(video_paths)}')
        parse_video(video_path, num_workers)


def find_max_obj(json_path, begin_frame=0, end_frame=None):
    """

    Examples:
        >>> find_max_obj(
        ...     r'H:\data/reolink\test\20251218\1\diff_peo\day\c15_fn\b26\RecM0A_20251217_161313_161812_0_531ED300000000_4D897C7.json',
        ...     begin_frame=1512,
        ...     end_frame=1911,
        ... )

    """
    ai_infos = read_json(json_path)
    areas = []
    for ai_info in ai_infos[begin_frame:end_frame]:
        for j, (cat, cat_dict) in enumerate(ai_info['action'].items()):
            if cat != 'pd':
                continue
            for obj in cat_dict['objs']:
                if obj['state'] != 1:
                    continue
                areas.append(obj['w'] * obj['h'])
    print(f'{len(areas) = }')
    print(f'{max(areas) = }')
    print(f'{sum(areas) / len(areas) = }')
    print(f'{min(areas) = }')



def main():
    # parse_video(
    #     r'H:\data/reolink\test\20251218\1\diff_peo\day\c15_fn\b26\RecM0A_20251218_091001_091500_0_531ED300000000_4D7661E.mp4')
    # find_max_obj(
    #     r'H:\data/reolink\test\20251218\1\diff_peo\day\c15_fn\b26\RecM0A_20251218_070001_070500_0_531ED300000000_4D53693.json',
    #     begin_frame=584,
    #     end_frame=688,
    # )
    parse_videos(r'H:\data\reolink\test\20251218\1\diff_peo\night\c15_fn\b26', 8)


if __name__ == '__main__':
    main()
