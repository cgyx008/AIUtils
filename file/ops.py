import hashlib
import os
import re
import time
from datetime import datetime
from pathlib import Path
import shutil

from tqdm import tqdm


def get_datetime_str():
    # time.strftime('%Y%m%d_%H%M%S', time.localtime())
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def cp(src, dst, glob_patten='**/*', exclude_dir='/None/', overwrite=False,
       map_txt=None, show_pbar=True, hard_link=False):
    """Copy src to dst (dir).

    Args:
        src (str | Path): a source directory or a glob path.
        dst (str | Path): destination directory, if not exists, create.
        glob_patten (str): global patten matching.
        exclude_dir (str): exclude directory.
        overwrite (bool): whether to overwrite old files
        map_txt (None | str | bool): whether to writedown src_to_dst.txt
        show_pbar (bool): Whether to show progress bar.
        hard_link (bool): Whether to use hard link.

    Examples:
        >>> cp('/home/kemove/218Algo/ganhao/AD/wd/v04/labels_add_vehicle_labels',
        ...    '/home/kemove/218Algo/ganhao/AD/wd/v04/labels_add_vehicle_labels_voc',
        ...    '**/*.xml',
        ...    'None')
    """
    if show_pbar:
        print(f'Copying {src} to {dst}')
    src, dst = Path(src), Path(dst)

    if map_txt:
        with open(map_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        paths = [line.strip().split(' -> ') for line in tqdm(lines)]
        src_files, dst_files = zip(*tqdm(paths))
        src_files = list(map(Path, tqdm(src_files)))
        dst_files = list(map(Path, tqdm(dst_files)))
    else:
        if src.is_file():
            src_files = [src]
        else:
            src_files = [p for p in src.glob(glob_patten)
                         if exclude_dir not in str(p)
                            and p.is_file()
                            and '\\' not in p.as_posix()]
            src_files.sort()
            # src_files = [Path(p.as_posix().replace('\\', '/'))
            #              for p in src_files]

        # Make destination parents
        if dst.suffix:
            dst_files = [dst]
            dst = dst.parent
        else:
            dst_files = [dst / p.relative_to(src) for p in tqdm(src_files)]
        create_parent_dirs(dst_files, show_pbar=show_pbar)

    if map_txt is None:
        with open(dst / 'src_to_dst.txt', 'w', encoding='utf-8') as f:
            for src_file, dst_file in zip(src_files, dst_files):
                f.write(f'{src_file.as_posix()} -> {dst_file.as_posix()}\n')

    # Copy files
    src_dst = list(zip(src_files, dst_files))
    pbar = show_pbar and tqdm(total=len(src_dst), ascii=True, smoothing=0)
    while src_dst:
        s, d = src_dst[0]

        try:
            # if (not overwrite
            #         and d.exists()
            #         and s.stat().st_size == d.stat().st_size):
            if d.exists() and s.stat().st_size == d.stat().st_size:
                src_dst.pop(0)
                if show_pbar:
                    pbar.update(1)
                continue

            if hard_link:
                d.hardlink_to(s)
            else:
                shutil.copy(s, d)
            # if s.stat().st_size == d.stat().st_size:
            #     src_dst.pop(0)
            #     pbar.update(1)

        except (OSError, BlockingIOError) as e:
            ts = get_datetime_str()
            print(f'\n{ts}: {s} -> {d}')
            print(e)
            time.sleep(1)
        except (Exception, KeyboardInterrupt) as e:
            ts = get_datetime_str()
            print(f'\n{ts}: {s} -> {d}')
            with open(dst / f'src_to_dst_{ts}.txt', 'w', encoding='utf-8') as f:
                for src_file, dst_file in src_dst:
                    f.write(f'{src_file.as_posix()} -> {dst_file.as_posix()}\n')
            raise e


def divide_dirs(root, num_divided_files=1000):
    """
    Divide files into different directories, {num_split_files} files in each directory.
    Args:
        root (str | Path): root
        num_divided_files (int): the number of  files in each directory
    Returns:

    """
    root = Path(root)
    paths = sorted(root.glob('*'))
    num_0s = len(str(len(paths) // num_divided_files))

    # Make directories
    parents = {str(root / str(i // num_divided_files).zfill(num_0s))
               for i in range(len(paths))}
    parents = sorted(parents)
    for parent in tqdm(parents):
        Path(parent).mkdir(parents=True, exist_ok=True)

    # Move files
    for i, p in enumerate(tqdm(paths, ascii=True)):
        shutil.move(p, root / str(i // num_divided_files).zfill(num_0s))


def merge_divided_dirs(root):
    """
    Merge divided directories.
    Args:
        root (str | Path): root
    """
    # Glob paths
    root = Path(root)
    paths = sorted(root.glob('**/*'))

    # Move files
    file_paths = [p for p in tqdm(paths) if p.is_file()]
    for p in tqdm(file_paths):
        shutil.move(p, root)

    # Remove empty directories
    dir_paths = [p for p in tqdm(paths) if p.is_dir()]
    for p in tqdm(dir_paths):
        if p.exists():
            shutil.rmtree(p)


def format_stem(stem):
    new_stem = re.sub(r'[^a-zA-Z0-9]', '_', stem)
    new_stem = re.sub('_{2,}', '_', new_stem)
    new_stem = new_stem.strip('_')
    return new_stem


def get_mtime(filepath):
    p = Path(filepath)
    stat = p.stat()
    ts = datetime.fromtimestamp(stat.st_mtime)
    mtime = ts.strftime("%Y%m%d_%H%M%S_%f")
    return mtime


def format_filename(filepath, postfix_sha256_16=True, sha256=''):
    p = Path(filepath)
    if postfix_sha256_16:
        sha256 = sha256 or calc_hash(p)
        new_stem = format_stem(f'{p.stem}_{sha256[:8]}')
    else:
        new_stem = format_stem(p.stem)
    if not new_stem:
        sha256 = sha256 or calc_hash(p)
        new_stem = format_stem(f'{p.stem}_{sha256[:8]}')
    return p.with_stem(new_stem)


def format_filenames(dir_path):
    root = Path(dir_path)
    paths = sorted(p for p in root.glob('**/*') if p.is_file())
    for p in tqdm(paths):
        p.rename(format_filename(p))


def rm_dirs(root, only_remove_empty=False):
    root = Path(root)
    dirs = sorted(p for p in root.glob('*/*') if p.is_dir())
    dirs = [p for p in tqdm(dirs) if not os.listdir(p)] if only_remove_empty else dirs
    for d in tqdm(dirs):
        shutil.rmtree(d)


def create_parent_dirs(paths, show_pbar=True):
    parent_dirs = sorted({Path(p).parent.as_posix()
                          for p in (tqdm(paths) if show_pbar else paths)})
    os.umask(0)
    for parent_dir in (tqdm(parent_dirs) if show_pbar else parent_dirs):
        Path(parent_dir).mkdir(parents=True, exist_ok=True)


def calc_hash(file_path, chunk_size=8192, method='sha256'):
    """
        计算文件的 SHA256 哈希值

        参数:
            file_path: 文件的路径
            chunk_size: 每次读取的文件块大小，默认为 8KB
            method: sha256 or md5

        返回:
            文件的 SHA256 哈希字符串（32位十六进制）
        """
    if not Path(file_path).exists():
        return '0' * 64

    if method == 'sha256':
        m_hash = hashlib.sha256()
    else:
        m_hash = hashlib.md5()
    while True:
        try:
            with open(file_path, "rb") as f:  # 用二进制模式打开文件
                # 循环读取文件，直到文件结束
                while chunk := f.read(chunk_size):
                    m_hash.update(chunk)  # 更新哈希对象
            return m_hash.hexdigest()
        except (OSError, BlockingIOError) as e:
            ts = get_datetime_str()
            print(f'\n{ts}: {e}. {file_path}')
            time.sleep(1)


def glob_path(root, pattern='**/*', txt_path=None):
    os.umask(0)

    root = Path(root)
    paths = sorted(root.glob(pattern))

    if txt_path is None:
        txt_path = root / 'glob_path.txt'
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines([f'{p}\n' for p in tqdm(paths)])


def main():
    # cp('/home/ganhao/data/ppvpd',
    #    '/mnt/28Server/animal/ovd/data/ppvpd',
    #    map_txt='/mnt/28Server/animal/ovd/data/ppvpd/src_to_dst.txt')
    # cp(r'H:\data\ade20k\ADE20K_2021_17_01.zip',
    #    r'Y:\Public_Datasets\ADE20K\ADE20K_2021_17_01.zip')

    # cp(r'W:\data_raid0\ganhao\data\youtube\processed\v1.0.0\20250816\20250906_crowdsourcing\20250909_001_100k_imgs',
    #    r'\\192.168.2.8\研发-IT-TEST-Algo\OVD\20250909_001_100k_imgs')
    # cp('/data_raid0/ganhao/data/youtube/processed/v1.0.0/20250715/20250722_crowdsourcing/20250801_002_20k_imgs',
    #    '/mnt/28Server/animal/ovd/data/reolink/working/20250801_002_20k_imgs')
    # cp('/data_raid0/ganhao/data/youtube/processed/v1.0.0/20250715/20250722_crowdsourcing/20250807_003_100k_imgs',
    #    '/mnt/28Server/animal/ovd/data/reolink/working/20250807_003_100k_imgs')

    # rm_dirs(r'\\192.168.2.8\研发-IT-TEST-Algo\OVD\20250817_004_100k_imgs')
    # cp('/home/kemove/2TSSD/ganhao/Projects',
    #    '/home/kemove/28Server/animal/ovd/projects')
    # cp('/home/ganhao/data/ppvpd/processed/v1.1.0_cleaning/images',
    #    '/home/ganhao/data/ppvpd/processed/v1.1.0_cleaning/images_for_dedup',
    #    glob_patten='*.jpg')
    # print(calc_sha256('/data_raid0/ganhao/data/ovd/inat/cougar/img_paths.txt'))
    # print(calc_hash('/home/ganhao/data/ovd/flickr30k/images/36979.jpg', method='md5'))
    # cp(
    #     '/home/ganhao/data/wr/inat',
    #     '/home/ganhao/data/inat',
    #     hard_link=True
    # )
    glob_path(
        '/data_raid0/ganhao/data/wd/test/20260310_youtube_bird',
        pattern='videos/*.mp4',
        txt_path='/data_raid0/ganhao/data/wd/test/20260310_youtube_bird/metadata/videos.txt'
    )


if __name__ == '__main__':
    main()
