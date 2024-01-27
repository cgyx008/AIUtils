import os
from concurrent.futures import ThreadPoolExecutor
# from functools import partial
from pathlib import Path

import cv2
from tqdm import trange, tqdm

from file.mv import get_time_prefix, format_stem


def decode_fourcc(cc):
    # 将视频格式数字解码为字符串
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def get_cap_and_attr(video_path):
    """
    读取视频和属性
    Args:
        video_path (str | Path): 视频路径

    Returns:
        (cv2.VideoCapture, int, int, int, float, str):
            视频对象，帧宽，帧高，帧数，帧率，FOURCC
    """
    cap = cv2.VideoCapture(str(video_path))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

    print(f"{video_path}视频属性：")
    print(f"帧数：{num_frames}")
    print(f"宽高：{width}， {height}")
    print(f"帧率：{fps}")
    print(f"格式：{fourcc}")

    return cap, width, height, num_frames, fps, fourcc


def extract_frames(video_path, steps=10, seconds=0, max_workers=8, ext='jpg',
                   extract_frames=False):
    """
    每{steps}帧提取1帧，并保存在和视频同名的文件夹中。
    Args:
        video_path (str | Path): 带后缀的视频名，如“D:/001.mp4”
        steps (int): 每{steps}帧提取1帧，默认为10，当为0时，按秒取帧
        seconds (float): 每{seconds}秒提取1帧，默认为0，表示按帧间隔取帧
        max_workers (int): 最大线程数。Windows在网络挂载硬盘使用多线程会占用大量内存，
            建议先在本地提帧，再复制到网络硬盘
        ext (str): 图片后缀，默认为jpg
        extract_frames (bool): 是否提取每一帧到frames文件夹
    """
    # 1. 读取视频和打印属性
    video_path = Path(video_path)
    cap, width, height, num_frames, fps, fourcc = get_cap_and_attr(video_path)

    # 2. 新建保存帧的文件夹，与视频同目录
    frames_dir = video_path.parent / video_path.stem / 'frames'
    images_dir = video_path.parent / video_path.stem / 'images'
    os.umask(0)
    frames_dir.mkdir(exist_ok=True, parents=True)
    images_dir.mkdir(exist_ok=True, parents=True)
    print(f'帧保存在文件夹：{frames_dir}')
    print(f'图片保存在文件夹：{images_dir}')

    # 如果i整除interval不等于0，跳过。每interval帧保存1帧。
    interval = steps or int(fps * seconds)
    if interval == 0:
        return

    # 3. 创建线程池
    executor = ThreadPoolExecutor(max_workers) if max_workers else None

    # 4. 提取帧
    # len(str(num_frames))自动计算需要填充多少个0。
    # 例如：视频有100帧，即num_frames=100，那么str(num_frames)='100'，
    #      len(str(num_frames))=3，所以需要填充3个0。
    num_0s = len(str(num_frames))
    for i in trange(num_frames, ascii=True):
        rtn, frame = cap.read()
        if not rtn:
            break

        # 图片名：视频名_帧索引.ext
        save_name = f'{video_path.stem}_{str(i).zfill(num_0s)}.{ext}'
        if extract_frames:
            save_path = frames_dir / save_name
            if executor:
                executor.submit(cv2.imwrite, str(save_path), frame)  # noqa
            else:
                cv2.imwrite(str(save_path), frame)

        # 如果i整除interval不等于0，跳过。每interval帧保存1帧。
        if i % interval != 0:
            continue

        save_path = images_dir / save_name
        if save_path.exists():
            continue
        if executor:
            executor.submit(cv2.imwrite, str(save_path), frame)  # noqa
        else:
            cv2.imwrite(str(save_path), frame)

    if executor:
        executor.shutdown()


def rewrite_video():
    video_path = Path(r'G:\Data\AD\reolink\test_feedback\20231125\TP_male_deer_FP_female_deer-RecM05_20231124_194143_194203_SR_671E8A000_487599.mp4')
    cap, width, height, num_frames, fps, fourcc = get_cap_and_attr(video_path)

    save_path = video_path.parent / f'{video_path.stem}_rewrite.mp4'
    vw = cv2.VideoWriter(str(save_path),
                         cv2.VideoWriter_fourcc(*'mp4v'), fps,
                         (width, height))
    for _ in trange(num_frames):
        _, frame = cap.read()
        vw.write(frame)

    cap.release()
    vw.release()


def extract_videos_in_a_dir():
    r = Path(r'T:\Private\Reolink\embedded_feedback\20240109')
    vs = sorted(r.glob('**/*.[am][pokv][4iv]'))
    # vs = sorted(r.glob('**/*.mp4'))
    # vs = sorted(p for p in vs if not (p.parent / p.stem).exists())
    print(f'Number of videos: {len(vs)}')
    for i, p in enumerate(vs):
        if i < 0:
            continue
        print(f'{i + 1} / {len(vs)}')
        extract_frames(p, steps=0, seconds=0.5, max_workers=8,
                       extract_frames=True)

    # fast about 30%
    # func = partial(extract_frames, steps=0, seconds=2, max_workers=0)
    # with ThreadPoolExecutor(8) as executor:
    #     list(executor.map(func, vs))


def rename_video():
    video_dir = Path(r'T:\Private\Reolink\embedded_feedback\20240109')
    video_paths = sorted(video_dir.glob('*.m[po][4v]'))
    path_map = {}
    for p in tqdm(video_paths):
        time_prefix = get_time_prefix(p)
        new_stem = format_stem(p.stem)
        new_stem = f'{time_prefix}_{new_stem}'
        new_stem = new_stem.strip('_')

        if new_stem in path_map:
            raise RuntimeError(f'Duplicate names: {path_map[new_stem]} and {p}')
        path_map[new_stem] = p

    for new_stem, p in tqdm(path_map.items()):
        new_path = p.with_stem(new_stem)
        p.rename(new_path)


def main():
    # extract_frames(r'D:\Projects\AD_pytools\AWS\Docker_wr_dev\test_crop\1440p_3_deer.mp4')
    # rename_video()
    extract_videos_in_a_dir()


if __name__ == '__main__':
    main()
