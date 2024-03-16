import datetime
import os
import re
from pathlib import Path
import shutil

from tqdm import tqdm, trange


def mv(src, dst, glob_patten='**/*', exclude_dir='/None/'):
    """Move src to dst (dir).

    Args:
        src (str | Path): a source directory or a glob path.
        dst (str | Path): destination directory, if not exists, create.
        glob_patten (str): global patten matching.
        exclude_dir (str): exclude directory.

    Examples:
        >>> mv('/home/kemove/218Algo/ganhao/AD/wd/v04/labels_add_vehicle_labels',
        >>>    '/home/kemove/218Algo/ganhao/AD/wd/v04/labels_add_vehicle_labels_voc',
        >>>     '**/*.xml',
        >>>     'None')
    """
    print(f'Copying {src} to {dst}')
    src, dst = Path(src), Path(dst)
    src_files = [p for p in src.glob(glob_patten)
                 if exclude_dir not in str(p) and p.is_file()]
    src_files.sort()

    # Make destination parents
    dst_parents = {dst / p.parent.relative_to(src) for p in src_files}
    for p in dst_parents:
        p.mkdir(parents=True, exist_ok=True)

    # Move files
    err_files = {}
    for p in tqdm(src_files, smoothing=0, ascii=True):
        if (dst / p.relative_to(src)).exists():
            continue
        # time.sleep(0.5)
        try:
            shutil.copy2(p, dst / p.relative_to(src))
        except OSError as e:
            print(e)
            print(f'src: {p}, dst: {dst / p.relative_to(src)}')
            err_files[str(p)] = dst / p.relative_to(src)
            continue

    if not err_files:
        return
    print('Copying error files...')
    for src, dst in tqdm(err_files.items(), smoothing=0, ascii=True):
        try:
            # time.sleep(0.5)
            shutil.copy2(src, dst)
        except OSError as e:
            print(e)
            print(f'src: {src}, dst: {dst}')
            continue


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
    for i, p in enumerate(tqdm(paths)):
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
        shutil.rmtree(p)


def format_stem(stem):
    new_stem = re.sub(r'[^a-zA-Z0-9]', '_', stem)
    new_stem = re.sub('_{2,}', '_', new_stem)
    new_stem = new_stem.strip('_')
    return new_stem


def get_time_prefix(filepath):
    p = Path(filepath)
    stat = p.stat()
    ts = datetime.datetime.fromtimestamp(stat.st_mtime)
    time_prefix = ts.strftime("%Y%m%d_%H%M%S_%f")
    return time_prefix


def format_filename(filepath):
    p = Path(filepath)
    new_stem = format_stem(p.stem) or get_time_prefix(filepath)
    return p.with_stem(new_stem)


def format_filenames(dir_path):
    root = Path(dir_path)
    paths = sorted(p for p in root.glob('**/*') if p.is_file())
    for p in tqdm(paths):
        p.rename(format_filename(p))


def mv_pred_label_dir():
    """
    Move predicted label directory to the same level as images directory
    """
    root = Path(r'T:\Private\Reolink\test_feedback\20240308')
    label_dirs = sorted(root.glob('**/predict/**/labels_xml'))
    for label_dir in tqdm(label_dirs):
        shutil.move(label_dir, label_dir.parents[1])


def rm_empty_dir():
    root = Path(r'T:\Private\Reolink\test_feedback\20240308')
    dirs = sorted(p for p in root.glob('**/predict') if p.is_dir())
    empty_dirs = [p for p in tqdm(dirs) if not os.listdir(p)]
    for d in tqdm(empty_dirs):
        shutil.rmtree(d)


def main():
    rm_empty_dir()


if __name__ == '__main__':
    main()
