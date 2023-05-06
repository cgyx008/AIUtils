from pathlib import Path
import shutil

from tqdm import tqdm


def mv(src, dst, glob_patten='**/*', exclude_dir='/wd/'):
    """Move src to dst (dir).

    Args:
        src (str | Path): a source directory or a glob path.
        dst (str | Path): destination directory, if not exists, create.
        glob_patten (str): global patten matching.
        exclude_dir (str): exclude directory.
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
    for p in tqdm(src_files, smoothing=0):
        shutil.move(p, dst / p.relative_to(src))


def main():
    mv('/home/kemove/4Tdisk/ganhao/AD',
       '/home/kemove/4Tdisk/ganhao/Data/AD')


if __name__ == '__main__':
    main()
