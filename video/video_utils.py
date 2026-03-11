import datetime
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from torchcodec.decoders import VideoDecoder
from torchvision.io import write_jpeg
from tqdm import trange, tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from file.ops import get_mtime, format_stem, calc_hash, format_filename, cp


video_sufs = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm', '.ts'}


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
                  extract_all_frames=False, images_dir=''):
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
        images_dir (str | Path): 保存帧的文件夹
    """
    # 1. 读取视频和打印属性
    video_path = Path(video_path)
    cap, width, height, num_frames, fps, fourcc = get_cap_and_attr(video_path)

    if width * height < 640 * 360:
        return

    # 2. 新建保存帧的文件夹，与视频同目录
    frames_dir = video_path.parent / video_path.stem / 'frames'
    images_dir = images_dir or video_path.parent / video_path.stem / 'images'
    images_dir = Path(images_dir)
    os.umask(0)
    if extract_all_frames:
        frames_dir.mkdir(exist_ok=True, parents=True)
        print(f'帧保存在文件夹：{frames_dir}')
    images_dir.mkdir(exist_ok=True, parents=True)
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
    # save_stem = format_video_stem(video_path, '', False)
    save_stem = images_dir.parts[-1]

    for i in trange(num_frames, ascii=True):
        rtn, frame = cap.read()
        if not rtn:
            break

        # 图片名：视频名_帧索引.ext
        save_name = f'{save_stem}_{str(i).zfill(num_0s)}.{ext}'
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


def extract_videos(src, dst='', video_src_root=''):
    # r = Path(r'H:\data\reolink\user\20241210')
    src = Path(src)
    suf = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm', '.ts'}
    if src.suffix == '.txt':
        with open(src, 'r', encoding='utf-8') as f:
            vs = [Path(line.strip()) for line in f.readlines()]
    elif src.suffix == '.json':
        with open(src, 'r', encoding='utf-8') as f:
            sha256_to_paths = json.load(f)
        vs = [paths[0] for paths in sha256_to_paths.values()]
    else:
        filepaths = list(src.glob('**/*.*'))
        vs = sorted([p for p in filepaths if p.suffix.lower() in suf])
    print(f'Number of videos: {len(vs)}')

    dst = Path(dst)
    video_src_root = Path(video_src_root)

    # extract_dir_to_path = {}
    # with open(src.parent / 'extract_dirs.csv', 'w', encoding='utf-8') as f:
    #     f.write(f'Video Path,Extract Directory\n')

    for i, p in enumerate(tqdm(vs)):
        # p, new_p = Path(p[0]), Path(p[1])
        print(f'{i + 1} / {len(vs)}')
        # if p.stem != 'Meteorite_crash_landing_captured_on_Ring_doorbell':
        #     continue
        if not p.exists():
            continue
        # video_stem = format_video_stem(p, data_prefix='', use_time_prefix=False)
        video_stem = p.stem
        # video_dir = (dst / p.relative_to(video_src_root)).parent / video_stem
        video_dir = dst / video_stem

        # if video_dir.as_posix in extract_dir_to_path or video_dir.exists():
        # if video_dir.as_posix() in extract_dir_to_path:
        #     video_stem = (video_stem + '_' + get_mtime(p)).strip('_')
        #     video_dir = video_dir.with_stem(video_stem)
        #
        #     # if video_dir.as_posix in extract_dir_to_path or video_dir.exists():
        #     if video_dir.as_posix() in extract_dir_to_path:
        #         with open(dst / 'duplicate_videos.txt', 'a', encoding='utf-8') as f:
        #             f.write(f'{p}\n')
        #         continue

        # extract_dir_to_path[video_dir.as_posix()] = p.as_posix()
        # with open(src.parent / 'extract_dirs.csv', 'a', encoding='utf-8') as f:
        #     f.write(f'{p.as_posix()},{video_dir.as_posix()}\n')

        # imgs_dir = video_dir / 'images'
        imgs_dir = video_dir

        extract_video_by_torch(p, save_dir=imgs_dir, verbose=False)
        # extract_video(p, steps=0, seconds=1, max_workers=8,
        #               extract_all_frames=False, images_dir=imgs_dir)
        print()


    # fast about 30%
    # func = partial(extract_frames, steps=0, seconds=2, max_workers=0)
    # with ThreadPoolExecutor(8) as executor:
    #     list(executor.map(func, vs))

def extract_videos_with_sha256_json(sha256_json, img_root):
    os.umask(0)

    img_root = Path(img_root)

    with open(sha256_json, 'r', encoding='utf-8') as f:
        sha256_to_path_pairs = json.load(f)

    for path_pairs in tqdm(sha256_to_path_pairs.values()):
        video_path = Path(path_pairs[0][1])  # 0: dedup, 1: rename path
        # img_dir = img_root / video_path.stem / 'images'
        img_dir = img_root / video_path.stem
        img_dir.mkdir(parents=True, exist_ok=True)
        extract_video(video_path, steps=0, seconds=1, max_workers=8,
                      extract_all_frames=False, images_dir=img_dir)



def format_video_stem(video_path, sha256_16_postfix=True,
                      data_prefix='', use_time_prefix=False):
    video_path = Path(video_path)
    if sha256_16_postfix:
        sha256 = calc_hash(video_path)
        new_stem = format_stem(f'{data_prefix}_{video_path.stem}_{sha256[:16]}')
    else:
        new_stem = format_stem(f'{data_prefix}_{video_path.stem}')
    # if use_time_prefix or not new_stem:
    #     time_prefix = get_mtime(video_path)
    #     new_stem = format_stem(f'{data_prefix}_{time_prefix}_{video_path.stem}')
    # time_prefix = get_time_prefix(video_path) if use_time_prefix else ''
    # new_stem = format_stem(f'{data_prefix}_{time_prefix}_{video_path.stem}')
    new_stem = new_stem.strip('_')
    return new_stem


def make_rename_video_map(video_dir):
    data_prefix = ''
    use_time_prefix = False
    # video_dir = Path(r'H:\data\wd\v009\20250226')
    suf = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm'}
    video_dir = Path(video_dir)
    video_paths = sorted([p for p in video_dir.glob('**/*.*')
                          if p.suffix.lower() in suf])
    path_map = {}
    for p in tqdm(video_paths):
        new_stem = format_video_stem(p, data_prefix=data_prefix,
                                     use_time_prefix=use_time_prefix)

        if new_stem in path_map:
            print(path_map[new_stem], p)
            raise RuntimeError(f'Duplicate names: {path_map[new_stem]} and {p}')
        path_map[new_stem] = p

    map_json = video_dir.parent / 'rename_video_map.json'
    with open(map_json, 'w', encoding='utf-8') as f:
        json.dump({k: v.as_posix() for k, v in path_map.items()}, f, indent=4)


def rename_videos_with_map(map_json):
    with open(map_json, 'r', encoding='utf-8') as f:
        path_map = json.load(f)

    for new_stem, p in tqdm(path_map.items()):
        new_path = p.with_stem(new_stem)
        p.rename(new_path)


def rename_videos(video_dir):
    """Deprecated. Use `make_rename_video_map` and `rename_videos_with_map` instead"""
    data_prefix = ''
    use_time_prefix = False
    # video_dir = Path(r'H:\data\wd\v009\20250226')
    suf = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm'}
    video_dir = Path(video_dir)
    video_paths = sorted([p for p in video_dir.glob('**/*.*')
                          if p.suffix.lower() in suf])
    path_map = {}
    for p in tqdm(video_paths):
        new_stem = format_video_stem(p, data_prefix=data_prefix,
                                     use_time_prefix=use_time_prefix)

        if new_stem in path_map:
            print(path_map[new_stem], p)
            raise RuntimeError(f'Duplicate names: {path_map[new_stem]} and {p}')
        path_map[new_stem] = p

    for new_stem, p in tqdm(path_map.items()):
        new_path = p.with_stem(new_stem)
        p.rename(new_path)


def rename_videos_with_sha256_json(sha256_json):
    with open(sha256_json, 'r', encoding='utf-8') as f:
        sha256_to_path_pairs = json.load(f)

    for sha256, path_pairs in tqdm(sha256_to_path_pairs.items()):
        for old_path, new_path in path_pairs:
            Path(old_path).rename(new_path)


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


def collect_frames_from_same_video(img_dir):
    img_dir = Path(img_dir)
    img_paths = sorted(img_dir.glob('*.jpg'))
    vid2imgs = defaultdict(list)
    for img_path in tqdm(img_paths):
        vid_stem = img_path.stem.rsplit('_', 1)[0]
        vid2imgs[vid_stem].append(img_path)

    os.umask(0)
    for v in vid2imgs:
        (img_dir / v).mkdir(exist_ok=True)

    for vid, img_paths in tqdm(vid2imgs.items()):
        for p in img_paths:
            shutil.move(p, img_dir / vid)


def calculate_videos_sha256(video_txt, num_threads=8):
    video_txt = Path(video_txt)
    with open(video_txt, 'r', encoding='utf-8') as f:
        video_paths = [Path(line.strip()) for line in f]

    with ThreadPoolExecutor(num_threads) as executor:
        sha256s = list(tqdm(executor.map(calc_hash, video_paths),
                            total=len(video_paths),
                            smoothing=0))
    assert len(video_paths) == len(sha256s)

    sha256_to_paths = defaultdict(list)
    for orig_path, sha256 in zip(tqdm(video_paths), sha256s):
        new_path = format_filename(orig_path, sha256=sha256)
        sha256_to_paths[sha256].append(
            (orig_path.as_posix(), new_path.as_posix())
        )
    print(f'Number of original videos: {len(video_paths)}')
    print(f'Number of dedup videos: {len(sha256_to_paths)}')

    with open(video_txt.parent / 'sha256.json', 'w', encoding='utf-8') as f:
        json.dump(sha256_to_paths, f, indent=4)


def check_extract_dirs_exist(extract_dirs_csv):
    with open(extract_dirs_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    not_exists_dirs = []
    for line in tqdm(lines[1:]):
        video_path, extract_dir = line.strip().rsplit(',', maxsplit=1)
        if not Path(extract_dir).exists():
            not_exists_dirs.append(extract_dir)
            print(f'{video_path = }')
            print(f'{extract_dir = }')
            print('-')

    print(f'{len(not_exists_dirs) = }')


def rename_and_extract(video_txt):
    video_txt = Path(video_txt)

    calculate_videos_sha256(video_txt)

    rename_videos_with_sha256_json(video_txt.parent / 'sha256.json')

    extract_videos_with_sha256_json(video_txt.parent / 'sha256.json',
                                    video_txt.parents[1] / 'images')


def cp_and_rename(src_txt, hash_algo='md5'):
    """
    dataset/
    +-- metadata/
        +-- src.txt       # input
        +-- md5.jsonl  # output, {src: 'xxx.mp4', dst: 'xxx_md5[:12].mp4', 'md5': 'xxx'}
        +-- src2dst.json  # output, {md5_0: [{src: 'xxx.mp4', dst: 'xxx_md5[:12].mp4'}],
                                     md5_1: [{src: 'yyy.mp4', dst: 'yyy_md5[:12].mp4'}],}
    +-- videos/
        +-- xxx_md5[:12].mp4  # output

    Args:
        video_txt (str | Path): Video txt.

    """
    src_txt = Path(src_txt)
    with open(src_txt, 'r', encoding='utf-8') as f:
        src_paths = [Path(line.strip()) for line in f]

    root = src_txt.parents[1]

    # Use ffmpeg to calculate md5 without video metadata
    jsonl = root / f'metadata/{hash_algo}.jsonl'
    calculated_src = {}
    if jsonl.exists():
        with open(jsonl, 'r', encoding='utf-8') as f:
            calculated_src = {json.loads(line.strip())['src'] for line in f}
    src_paths = [p for p in src_paths if p.as_posix() not in calculated_src]

    for src in tqdm(src_paths):
        # if src.as_posix() in calculated_src:
        #     continue
        if not src.exists():
            continue
        cmd = ['ffmpeg', '-loglevel', 'error', '-i', src.as_posix(),
               '-map', '0:v:0', '-f', 'hash', '-hash', hash_algo, '-']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    check=True)

            h = result.stdout.strip().split('=')[1]
            new_stem = format_stem(f'{src.stem}_{h[:12]}')
            dst = root / f'videos/{new_stem}{src.suffix}'

            with open(jsonl, 'a', encoding='utf-8') as f:
                f.write(
                    json.dumps(
                        {'src': src.as_posix(),
                         'dst': dst.as_posix(),
                         hash_algo: h}
                    ) + '\n'
                )

        except subprocess.CalledProcessError as e:
            # print(f"⚠️ Skip the file! FFMPEG runs failed!")
            # print(f"Error file: {e.cmd[4]}")  # e.cmd[4] is usually the input file
            # print(f"Error: {e.stderr.strip()}")

            with open(root / 'metadata/md5_error.jsonl', 'a', encoding='utf-8') as f:
                f.write(
                    json.dumps({'src': src.as_posix(), 'error': e.stderr})
                    + '\n'
                )

            continue

    with open(jsonl, 'r', encoding='utf-8') as f:
        json_lines = [json.loads(line.strip()) for line in f]
    h_to_paths = {}
    for line in json_lines:
        h = line[hash_algo]
        if line[hash_algo] not in h_to_paths:
            h_to_paths[h] = [{'src': line['src'], 'dst': line['dst']}]
        else:
            h_to_paths[h].append({'src': line['src'], 'dst': line['dst']})
    with open(root / 'metadata/src2dst.json', 'w', encoding='utf-8') as f:
        json.dump(h_to_paths, f, ensure_ascii=False, indent=4)

    for paths in tqdm(h_to_paths.values()):
        src, dst = paths[0]['src'], paths[0]['dst']
        cp(src, dst, map_txt=False, show_pbar=False)


def calc_stream_md5_by_ffmpeg(video_path):
    """Calculate video stream md5 by ffmpeg without video container."""
    video_path = Path(video_path)

    if not video_path.exists():
        return {'md5': '0' * 32,
                'err': f'File not exists: {video_path.as_posix()}'}

    cmd = ['ffmpeg', '-loglevel', 'error', '-i', video_path.as_posix(),
           '-map', '0:v:0', '-f', 'hash', '-hash', 'md5', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True)
        h = result.stdout.strip().split('=')[1]
        return {'md5': h, 'err': None}

    except subprocess.CalledProcessError as e:
        # print(f"⚠️ Skip the file! FFMPEG runs failed!")
        # print(f"Error file: {e.cmd[4]}")  # e.cmd[4] is usually the input file
        # print(f"Error: {e.stderr.strip()}")

        return {'md5': '0' * 32, 'err': e.stderr}


def extract_video_by_torch(video_path, save_dir='', step_seconds=1, verbose=True):
    """
    Use torchcodec to extract video frames, because opencv has no av1 decoder.
    Args:
        video_path (str | Path): Video path.
        save_dir (str | Path): Save directory.
        step_seconds (int | float): Each {step_seconds} seconds extract 1 frame.
        verbose (bool): Whether to print metadata and show progress bar.
    """
    os.umask(0)

    video_path = Path(video_path)
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=8)

    print(video_path)
    print(decoder.metadata)
    # VideoStreamMetadata:
    #   duration_seconds_from_header: 13.8
    #   begin_stream_seconds_from_header: 0.0
    #   bit_rate: 505790.0
    #   codec: h264
    #   stream_index: 0
    #   duration_seconds: 13.8
    #   begin_stream_seconds: 0.0
    #   begin_stream_seconds_from_content: 0.0
    #   end_stream_seconds_from_content: 13.8
    #   width: 640
    #   height: 360
    #   num_frames_from_header: 345
    #   num_frames_from_content: 345
    #   average_fps_from_header: 25.0
    #   pixel_aspect_ratio: 1
    #   end_stream_seconds: 13.8
    #   num_frames: 345
    #   average_fps: 25.0

    if decoder.metadata.width * decoder.metadata.height < 640 * 360:
        return

    target_fps = round(decoder.metadata.average_fps * step_seconds)
    frame_ids = list(range(0, decoder.metadata.num_frames, target_fps))
    frames = decoder.get_frames_at(frame_ids)

    save_dir = save_dir or video_path.parent / video_path.stem
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_0s = len(str(frame_ids[-1]))

    iters = zip(tqdm(frame_ids), frames) if verbose else zip(frame_ids, frames)
    for i, frame in iters:
        save_path = save_dir / f'{video_path.stem}_{str(i).zfill(num_0s)}.jpg'
        write_jpeg(frame.data, save_path.as_posix())


class VideoDataset:
    """
    VideoDataset folder structure:
    data_root/
    +-- raw/
        +-- 001.mp4
    +-- videos/
        +-- 001_md5.mp4
    +-- metadata/
    +-- images/
        +-- 001_md5/
            +-- 001_md5_000.jpg
    """
    def __init__(self, root):
        os.umask(0)

        self.root = Path(root)
        self.mkdirs()
        self.raw_txt = self.make_raw_txt()
        self.md5_jsonl = self.calc_md5()
        self.video_txt = self.cp_raw_to_videos_dir()
        self.extract_videos()

    def mkdirs(self):
        (self.root / 'metadata').mkdir(exist_ok=True)
        (self.root / 'videos').mkdir(exist_ok=True)
        (self.root / 'images').mkdir(exist_ok=True)

    def make_raw_txt(self):
        video_paths = [p for p in self.root.glob('raw/**/*')
                       if p.is_file() and p.suffix in video_sufs]
        video_paths.sort()
        raw_txt = self.root / 'metadata/raw.txt'
        with open(raw_txt, 'w', encoding='utf-8') as f:
            f.writelines([f'{p.as_posix()}\n' for p in video_paths])
        return raw_txt

    def calc_md5(self):
        with open(self.raw_txt, encoding='utf-8') as f:
            video_paths = [Path(line.strip()) for line in f]

        md5_jsonl = self.root / 'metadata/raw_md5.jsonl'
        err_jsonl = self.root / 'metadata/raw_md5_err.jsonl'

        if md5_jsonl.exists():
            with open(md5_jsonl, encoding='utf-8') as f:
                calc_src = {json.loads(line)['src'] for line in f}
            video_paths = [p for p in video_paths
                           if p.as_posix() not in calc_src]

        for video_path in tqdm(video_paths, desc='Calculating md5'):
            md5_res = calc_stream_md5_by_ffmpeg(video_path)
            if md5_res['err'] is None:
                with open(md5_jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'md5': md5_res['md5'],
                                        'src': video_path.as_posix()},
                                       ensure_ascii=False))
                    f.write('\n')
            else:
                with open(err_jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'err': md5_res['err'],
                                        'src': video_path.as_posix()}))
                    f.write('\n')
        return md5_jsonl

    def cp_raw_to_videos_dir(self):
        with open(self.md5_jsonl, encoding='utf-8') as f:
            raw_md5s = [json.loads(line) for line in f]

        md5_to_cp = defaultdict(list)
        for d in tqdm(raw_md5s):
            src, md5 = Path(d['src']), d['md5']
            new_stem = format_stem(f'{src.stem}_{md5[:12]}')
            dst = self.root / f'videos/{new_stem}{src.suffix}'
            md5_to_cp[md5].append({'src': src.as_posix(), 'dst': dst.as_posix()})

        md5_to_cp_json = self.root / 'metadata/md5_to_cp.json'
        with open(md5_to_cp_json, 'w', encoding='utf-8') as f:
            json.dump(md5_to_cp, f, ensure_ascii=False, indent=4)

        video_txt = self.root / 'metadata/videos.txt'
        with open(video_txt, 'w', encoding='utf-8') as f:
            f.writelines([cp[0]['dst'] + '\n' for cp in md5_to_cp.values()])

        for cp in md5_to_cp.values():
            src, dst = Path(cp[0]['src']), Path(cp[0]['dst'])
            dst.hardlink_to(src)

        return video_txt

    def extract_videos(self):
        with open(self.video_txt, encoding='utf-8') as f:
            video_paths = [Path(line.strip()) for line in f]

        for video_path in tqdm(video_paths):
            extract_video_by_torch(
                video_path,
                save_dir=self.root / f'images/{video_path.stem}',
                verbose=False,
            )

        img_paths = sorted(self.root.glob('images/*/*.jpg'))
        with open(self.root / 'metadata/images.txt', 'w', encoding='utf-8') as f:
            f.writelines([f'{p.as_posix()}\n' for p in img_paths])


def main():
    # extract_videos(
    #     '/data_raid0/ganhao/data/youtube/processed/v1.0.0/20250908/sha256.json',
    #     dst='/data_raid0/ganhao/data/youtube/processed/v1.0.0/20250908/images',
    #     video_src_root='/mnt/28Server/common/AlgoTestVideos/Youtube',
    # )
    # check_extract_dirs_exist(
    #     '/data_raid0/ganhao/data/youtube/processed/v1.0.0/20250816/extract_dirs.csv'
    # )
    # make_rename_video_map('/data/ganhao/data/ovd/web/20260109_garage_door/videos')
    # extract_video_by_torch(
    #     r'/mnt/28Server/animal/ovd/data/youtube/20260226_package/videos/Amazon_delivery_guy_thank_you_bro_shortvideo_automobile_shots_funny_a757d4d7d081.mp4',
    #     save_dir='/mnt/28Server/animal/ovd/data/youtube/20260226_package/debug/Amazon_delivery_guy_thank_you_bro_shortvideo_automobile_shots_funny_a757d4d7d081'
    # )
    # rename_and_extract('')
    # cp_and_rename('/mnt/28Server/animal/ovd/data/youtube/20260226_package/metadata/src.txt')
    # extract_videos(
    #     '/mnt/28Server/animal/ovd/data/youtube/20260226_package/metadata/empty_videos.txt',
    #     dst='/mnt/28Server/animal/ovd/data/youtube/20260226_package/empty',
    # )
    VideoDataset('/home/ganhao/data/wr/src/20260311_youtube_animal')


if __name__ == '__main__':
    main()
