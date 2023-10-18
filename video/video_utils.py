import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import trange


def decode_fourcc(cc):
    # 将视频格式数字解码为字符串
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def get_cap_and_attr(video_path):
    """
    读取视频和属性
    Args:
        video_path (str | Path): 视频路径

    Returns:
        (cv2.VideoCapture, int, int, int, str, str): 视频对象，帧数，帧宽，帧高，帧率，FOURCC
    """
    cap = cv2.VideoCapture(str(video_path))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

    print(f"{video_path}视频属性：")
    print(f"帧数：{num_frames}")
    print(f"宽高：{width}， {height}")
    print(f"帧率：{fps}")
    print(f"格式：{fourcc}")

    return cap, num_frames, width, height, fps, fourcc


def extract_frames(video_path, steps=10, max_workers=0, ext='jpg'):
    """
    每{steps}帧提取1帧，并保存在和视频同名的文件夹中。
    Args:
        video_path (str | Path): 带后缀的视频名，如“D:/001.mp4”
        steps (int): 每{steps}帧提取1帧，默认为10
        max_workers (int): 最大线程数
        ext (str): 图片后缀，默认为jpg
    """
    # 1. 读取视频和打印属性
    video_path = Path(video_path)
    cap, num_frames = get_cap_and_attr(video_path)[:2]

    # 2. 新建保存帧的文件夹，与视频同目录
    frames_dir = video_path.parent / video_path.stem / 'frames'
    images_dir = video_path.parent / video_path.stem / 'images'
    os.umask(0)
    frames_dir.mkdir(exist_ok=True, parents=True)
    images_dir.mkdir(exist_ok=True, parents=True)
    print(f'帧保存在文件夹：{frames_dir}')
    print(f'图片保存在文件夹：{images_dir}')

    # 3. 创建线程池
    executor = ThreadPoolExecutor(max_workers) if max_workers else None

    # 4. 提取帧
    # len(str(num_frames))自动计算需要填充多少个0。
    # 例如：视频有100帧，即num_frames=100，那么str(num_frames)='100'，
    #      len(str(num_frames))=3，所以需要填充3个0。
    num_0s = len(str(num_frames))
    for i in trange(num_frames):
        rtn, frame = cap.read()
        if not rtn:
            break

        # 图片名：视频名_帧索引.ext
        save_name = f'{video_path.stem}_{str(i).zfill(num_0s)}.{ext}'
        save_path = frames_dir / save_name
        if executor:
            executor.submit(cv2.imwrite, str(save_path), frame)  # noqa
        else:
            cv2.imwrite(str(save_path), frame)

        # 如果i整除steps不等于0，跳过。每steps帧保存1帧。
        if i % steps != 0:
            continue

        save_path = images_dir / save_name
        if executor:
            executor.submit(cv2.imwrite, str(save_path), frame)  # noqa
        else:
            cv2.imwrite(str(save_path), frame)

    if executor:
        executor.shutdown()


def rewrite_video():
    video_path = Path(r'G:\Data\AD\reolink\videos\EnWeChat\antelope_fp_person\fn_dog.mp4')
    cap, num_frames, width, height, fps, _ = get_cap_and_attr(video_path)

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
    r = Path(r'T:\Private\Reolink\embedded_feedback')
    # vs = sorted(r.glob('*.m[pok][4v]'))
    vs = sorted(r.glob('**/*.mp4'))
    # vs = sorted(p for p in vs if not (p.parent / p.stem).exists())
    print(f'Number of videos: {len(vs)}')
    for i, p in enumerate(vs):
        print(f'{i + 1} / {len(vs)}')
        extract_frames(p)


def main():
    extract_videos_in_a_dir()


if __name__ == '__main__':
    main()
