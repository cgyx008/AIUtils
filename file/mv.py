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
    src, dst = Path(src), Path(dst)
    src_files = [p for p in src.glob(glob_patten)
                 if exclude_dir not in str(p) and p.is_file()]
    src_files.sort()

    # Make destination parents
    dst_parents = {dst / p.parent.relative_to(src) for p in src_files}
    for p in dst_parents:
        p.mkdir(parents=True, exist_ok=True)

    # Move files
    for p in tqdm(src_files, smoothing=0, ascii=True):
        shutil.copy2(p, dst / p.relative_to(src))


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
    for i in trange(len(paths) // num_divided_files):
        (root / str(i).zfill(num_0s)).mkdir(parents=True, exist_ok=True)

    # Move files
    for i, p in enumerate(tqdm(paths)):
        shutil.move(p, root / str(i).zfill(num_0s))


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


def main():
    mv(
        r'T:\Working\v05\add_test_feedback\train',
        r'T:\Train\v05\train',
        exclude_dir='labels_xml'
    )


if __name__ == '__main__':
    main()
