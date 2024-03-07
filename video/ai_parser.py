import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())
from image.image_utils import draw_rect_and_put_text
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
            if w == 1 and h == 1:
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


def parse_video(video_path, num_workers=8):
    video_path = Path(video_path)

    model_results = get_model_results(video_path)
    action_results = get_action_results(video_path)

    cap, width, height, num_frames, *_ = get_cap_and_attr(video_path)
    num_0s = len(str(num_frames))
    save_dir = video_path.parent / video_path.stem / 'model_and_action'
    action_dir = video_path.parent / video_path.stem / 'action'
    save_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    colors = {'ad': (255, 0, 0), 'pd': (0, 255, 0), 'vd': (0, 0, 255)}
    executor = ThreadPoolExecutor(num_workers) if num_workers else None
    for i, (mo, ao) in enumerate(zip(tqdm(model_results), action_results)):
        ret, model_frame = cap.read()
        action_frame = model_frame.copy()
        action = False
        for cls, outputs in model_results[mo].items():
            color = colors[cls]
            for *box, score in outputs:
                box[0] *= width
                box[1] *= height
                box[2] *= width
                box[3] *= height
                text = f'{cls} {score}'
                draw_rect_and_put_text(model_frame, box, text, color, 2)

        for cls, outputs in action_results[ao].items():
            color = colors[cls]
            for *box, score, rid, state in outputs:
                if state != 1:
                    continue
                action = True
                box[0] *= width
                box[1] *= height
                box[2] *= width
                box[3] *= height
                text = f'{cls} {score}'
                draw_rect_and_put_text(action_frame, box, text, color, 2)

        img = cv2.hconcat([model_frame, action_frame])
        save_name = f'{video_path.stem}_{str(i).zfill(num_0s)}.jpg'
        save_path = save_dir / save_name
        if executor is not None:
            executor.submit(cv2.imwrite, str(save_path), img)  # noqa
        else:
            cv2.imwrite(save_path.as_posix(), img)
        if action:
            save_path = action_dir / save_name
            if executor is not None:
                executor.submit(cv2.imwrite, str(save_path), img)  # noqa
            else:
                cv2.imwrite(save_path.as_posix(), img)


def parse_videos(video_dir, num_workers=8):
    video_paths = sorted(Path(video_dir).glob('*.mp4'))
    for video_path in tqdm(video_paths):
        parse_video(video_path, num_workers)


def main():
    parse_videos(
        r'G:\data\fepvd\private\reolink\test_feedback\20240306',
        0
    )


if __name__ == '__main__':
    main()
