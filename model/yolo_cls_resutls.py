import shutil
from pathlib import Path

from tqdm import tqdm


def get_labels(img_dir):
    img_dir = Path(img_dir)
    img_paths = sorted(img_dir.glob('**/*.[jp][pn]g'))
    labels = {}
    for p in tqdm(img_paths):
        if p.stem not in labels:
            labels[p.stem] = p
        else:
            print(labels[p.stem])
            print(p)
    assert len(img_paths) == len(labels)
    return labels


def get_preds(txt_dir):
    txt_dir = Path(txt_dir)
    txt_paths = sorted(txt_dir.glob('*.txt'))
    print('Reading txts...')
    preds = {}
    for p in tqdm(txt_paths):
        with open(p, 'r', encoding='utf-8') as f:
            preds[p.stem] = f.readline().split()[1]
    return preds


def save_fp(img_dir, txt_dir, fp_dir):
    """
    保存YOLO误分图片。
    Args:
        img_dir (str | Path): 训练集或测试集文件夹
        txt_dir (str | Path): 预测的txt标签文件夹
        fp_dir (str | Path): 保存误分图片的文件夹

    Examples:
        >>> img_dir = 'G:/data/wr/v03/v03_15/test'
        >>> txt_dir = 'Z:/8TSSD/ganhao/projects/ultralytics/runs/classify/wr/predict/wr_v03_15_001_split_test_test/labels'
        >>> fp_dir = 'G:/data/wr/v03/v03_15/working/wr_v03_15_001_test'
        >>> save_fp(img_dir, txt_dir, fp_dir)
    """
    img_dir, txt_dir, fp_dir = Path(img_dir), Path(txt_dir), Path(fp_dir)

    labels = get_labels(img_dir)
    preds = get_preds(txt_dir)
    assert len(labels) == len(preds)

    for stem in tqdm(labels):
        img_path = labels[stem]
        label_cls = img_path.parts[len(img_dir.parts)]
        pred_cls = preds[stem]
        if label_cls == pred_cls:
            continue
        dst = fp_dir / label_cls / pred_cls
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst)


def main():
    save_fp(
        'G:/data/wr/v03/v03_15/test',
        'Z:/8TSSD/ganhao/projects/ultralytics/runs/classify/wr/predict/wr_v03_15_005_tune_000_scale_0_test/labels',
        'G:/data/wr/v03/v03_15/working/wr_v03_15_005_tune_000_scale_0_test'
    )


if __name__ == '__main__':
    main()
