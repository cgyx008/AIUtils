import datetime
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
# from functools import partial
from pathlib import Path

import cv2
from tqdm import trange, tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from file.ops import get_time_prefix, format_stem


def decode_fourcc(cc):
    # 将视频格式数字解码为字符串
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def get_cap_and_attr(video_path, verbose=True):
    """
    读取视频和属性
    Args:
        video_path (str | Path): 视频路径
        verbose (bool): 是否打印视频基本信息

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

    if verbose:
        print(f"{video_path}视频属性：")
        print(f"帧数：{num_frames}")
        print(f"宽高：{width}， {height}")
        print(f"帧率：{fps}")
        print(f"格式：{fourcc}")

    return cap, width, height, num_frames, fps, fourcc


def extract_video(video_path, steps=0, seconds=1, max_workers=8, ext='jpg',
                  extract_all_frames=False):
    """
    每{steps}帧提取1帧，并保存在和视频同名的文件夹中。
    Args:
        video_path (str | Path): 带后缀的视频名，如“D:/001.mp4”
        steps (int): 每{steps}帧提取1帧，默认为10，当为0时，按秒取帧
        seconds (float): 每{seconds}秒提取1帧，默认为0，表示按帧间隔取帧
        max_workers (int): 最大线程数。Windows在网络挂载硬盘使用多线程会占用大量内存，
            建议先在本地提帧，再复制到网络硬盘
        ext (str): 图片后缀，默认为jpg
        extract_all_frames (bool): 是否提取每一帧到frames文件夹
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

    if not cap.isOpened() or num_frames <= 0:
        return

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
        if extract_all_frames:
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


def rewrite_video(video_path):
    # video_path = Path(r'H:\data\test\20250118\910777916293978.mp4')
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


def extract_videos(r):
    # r = Path(r'H:\data\reolink\user\20241210')
    vs = sorted(r.glob('*.[amw][pokvm][4iv]'))
    # vs = sorted(r.glob('**/*.mp4'))
    # vs = sorted(p for p in vs if not (p.parent / p.stem).exists())
    print(f'Number of videos: {len(vs)}')
    for i, p in enumerate(vs):
        if i < 0:
            continue
        print(f'{i + 1} / {len(vs)}')
        extract_video(p, steps=0, seconds=1, max_workers=8,
                      extract_all_frames=False)

    # fast about 30%
    # func = partial(extract_frames, steps=0, seconds=2, max_workers=0)
    # with ThreadPoolExecutor(8) as executor:
    #     list(executor.map(func, vs))


def format_video_stem(video_path, data_prefix='', use_time_prefix=True):
    video_path = Path(video_path)
    time_prefix = get_time_prefix(video_path) if use_time_prefix else ''
    new_stem = format_stem(f'{data_prefix}_{time_prefix}_{video_path.stem}')
    new_stem = new_stem.strip('_')
    return new_stem


def rename_videos(video_dir):
    data_prefix = ''
    use_time_prefix = False
    # video_dir = Path(r'H:\data\wd\v009\20250226')
    video_paths = sorted(video_dir.glob('**/*.[amw][mopv][4iv]'))
    path_map = {}
    for p in tqdm(video_paths):
        new_stem = format_video_stem(p, data_prefix, use_time_prefix)

        if new_stem in path_map:
            print(path_map[new_stem], p)
            raise RuntimeError(f'Duplicate names: {path_map[new_stem]} and {p}')
        path_map[new_stem] = p

    for new_stem, p in tqdm(path_map.items()):
        new_path = p.with_stem(new_stem)
        p.rename(new_path)


def copy_videos(src_dir, dst_dir):
    roots = [Path(src_dir)]
    # dst_dir = Path(r'U:\Animal\Private\reolink\user_feedback')

    video_paths = []
    for root in roots:
        video_paths += list(root.glob('**/*.[am][opv][4iv]'))
    video_paths.sort()

    csv_path = dst_dir / 'video_copy_info_20240221.csv'
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write('Source,Destination,VideoID (mtime size)\n')

    vts = {}
    for video_path in tqdm(video_paths):
        stat = video_path.stat()
        ts = datetime.datetime.fromtimestamp(stat.st_mtime)
        video_id = f"{ts.strftime('%Y-%m-%d_%H-%M-%S.%f')} {stat.st_size}"
        if video_id not in vts:
            new_stem = format_video_stem(video_path)
            new_name = video_path.with_stem(new_stem).name
            dst = dst_dir / str(ts.year) / str(ts.month).zfill(2) / new_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            vts[video_id] = [(video_path, dst)]

        else:
            dst = vts[video_id][0][1]
            vts[video_id].append((video_path, dst))

        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(f'{video_path.as_posix()},{dst.as_posix()},{video_id}\n')


def gen_video_id(video_path):
    """
    Generate video id from the video content.
    Args:
        video_path (str | Path): video path

    Returns:
        (str): f'{frame_w}_{frame_h}_{num_frames}_{fps}_{size}_{frame_sum}'
    """
    size = Path(video_path).stat().st_size
    cap, width, height, num_frames, fps, _ = get_cap_and_attr(video_path, False)
    ret, frame = cap.read()
    frame_sum = frame.sum() if ret else 0
    frame_var = frame.var() if ret else 0
    return f'{width}_{height}_{num_frames}_{fps}_{size}_{frame_sum}_{frame_var}'


def get_video_ids(video_root):
    # video_root = Path('/mnt/28Server/common/AlgoTestVideos/OfficialWebsite')
    video_paths = sorted(video_root.glob('*.[amwAMW][mopvMOPV][4ivIV]'))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = video_root / f'video_ids_{ts}.csv'

    for video_path in tqdm(video_paths):
        video_id = gen_video_id(video_path)
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(f'{video_path},{video_id}\n')


def get_id2video(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    id2video = {}
    for line in tqdm(lines):
        video_path, video_id = line.strip().rsplit(',', 1)
        if video_id not in id2video:
            id2video[video_id] = [video_path]
        else:
            id2video[video_id].append(video_path)
    return id2video


def cmp_csv(old_csv, new_csv):
    id2video0 = get_id2video(old_csv)
    id2video1 = get_id2video(new_csv)
    old_videos, new_videos = defaultdict(list), defaultdict(list)
    for k, v in id2video1.items():
        if k in id2video0:
            old_videos[k].extend(v)
        else:
            new_videos[k].extend(v)
    return old_videos, new_videos


def rm_old_videos(old_csv, new_csv):
    old_videos, new_videos = cmp_csv(old_csv, new_csv)
    for v in old_videos.values():
        for p in v:
            Path(p).unlink()


def copy_new_videos(old_csv, new_csv, dst_dir):
    """
    csv format:
    video_path, f'{frame_w}_{frame_h}_{num_frames}_{fps}_{size}_{sum_frame_0}

    Args:
        old_csv (str|Path): csv file containing existing videos
        new_csv (str|Path): csv file containing existing and new videos
        dst_dir (str|Path): destination directory for saving new videos

    Examples:
        >>> root = 'U:/Animal/Private/reolink/user_feedback'
        >>> copy_new_videos(
        >>>    f'{root}/video_ids_20240222_172648.csv',
        >>>    f'{root}/video_ids_20240222_181440.csv',
        >>>    f'{root}/20240222'
        >>> )
    """

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    old_videos, new_videos = cmp_csv(old_csv, new_csv)

    name2video = {}
    for v in tqdm(new_videos.values()):
        names = [Path(p).with_stem(format_video_stem(p)).name for p in v]
        name = sorted(names)[0]
        if name not in name2video:
            name2video[name] = v
        else:
            name2video[name].extend(v)
            print(name2video[name])
    assert len(new_videos) == len(name2video)

    for k, v in tqdm(name2video.items()):
        shutil.copy2(v[0], dst_dir / k)


def loop_video_dirs(root, func, *args, **kwargs):
    root = Path(root)
    video_dirs = sorted(p for p in root.glob('*') if p.is_dir())
    for i, video_dir in enumerate(video_dirs):
        print(f'[{i + 1} / {len(video_dirs)}] {video_dir}')
        func(video_dir, *args, **kwargs)


def main():
    extract_videos(Path(''))


if __name__ == '__main__':
    main()
