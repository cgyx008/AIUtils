from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import trange


def decode_fourcc(cc):
    # 将视频格式数字解码为字符串
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def extract_frames(video_path, steps=10, ext='jpg', max_workers=8):
    """
    每{steps}帧提取1帧，并保存在和视频同名的文件夹中。
    Args:
        video_path (str): 带后缀的视频名，如“D:/001.mp4”
        steps (int): 每{steps}帧提取1帧，默认为10
        ext (str): 图片后缀，默认为jpg
        max_workers (int): 最大线程数
    """
    # 1. 读取视频
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    # 2. 打印视频属性
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    print(f'{video_path}视频属性：')
    print(f'帧数：{num_frames}')
    print(f'宽高：{width}， {height}')
    print(f'帧率：{fps}')
    print(f'格式：{fourcc}')

    # 3. 新建保存帧的文件夹，与视频同目录
    save_dir = video_path.parent / video_path.stem / 'images'
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f'帧保存在文件夹：{save_dir}')

    # 4. 创建线程池
    executor = ThreadPoolExecutor(max_workers)

    # 5. 提取帧
    # len(str(num_frames))自动计算需要填充多少个0。
    # 例如：视频有100帧，即num_frames=100，那么str(num_frames)='100'，
    #      len(str(num_frames))=3，所以需要填充3个0。
    num_0s = len(str(num_frames))
    # 图片名：视频名_帧索引.ext
    save_name = '{}_{:0>{}d}.{}'
    for i in trange(num_frames):
        rtn, frame = cap.read()
        if not rtn:
            break

        # 如果i整除steps不等于0，跳过。每steps帧保存1帧。
        if i % steps != 0:
            continue

        save_path = save_dir / save_name.format(video_path.stem, i, num_0s, ext)
        executor.submit(cv2.imwrite, str(save_path), frame)  # noqa


def rewrite_video():
    video_path = Path(r'G:\Data\AD\reolink\videos\EnWeChat\antelope_fp_person\fn_dog.mp4')
    cap = cv2.VideoCapture(str(video_path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    r = Path(r'G:\Data\FEPD\Reolink\0918_feedback')
    vs = sorted(r.glob('*.m[pok][4v]'))
    # vs = sorted(p for p in vs if not (p.parent / p.stem).exists())
    print(f'Number of videos: {len(vs)}')
    for i, p in enumerate(vs):
        print(f'{i + 1} / {len(vs)}')
        extract_frames(str(p), 1)


def main():
    extract_videos_in_a_dir()


if __name__ == '__main__':
    main()
