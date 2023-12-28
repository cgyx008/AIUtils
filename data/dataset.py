import random
from pathlib import Path


def split_train_val(img_root):
    img_root = Path(img_root)
    img_paths = sorted(img_root.glob('**/images/**/*.[jp][pn]g'))
    img_names = list({p.name for p in img_paths})
    assert len(img_paths) == len(img_names)
    random.shuffle(img_names)

    num_val = len(img_names) // 10
    train_img_stems = sorted(img_names[:-num_val])
    val_img_stems = sorted(img_names[-num_val:])

    with open(img_root / 'train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_img_stems) + '\n')
    with open(img_root / 'val.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_img_stems) + '\n')


def main():
    split_train_val(r'T:\Working\v05\add_test_feedback')


if __name__ == '__main__':
    main()
