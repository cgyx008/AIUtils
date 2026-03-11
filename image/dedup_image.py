import json
import os
import shutil
from pathlib import Path

from imagededup.methods import PHash
# from imagededup.utils import plot_duplicates
from tqdm import tqdm


def dedup(image_dir, copy=True):
    image_dir = Path(image_dir)
    phasher = PHash()

    # Generate encodings for all images in an image directory
    # encodings: dict[str, str] = {'img_001.jpg': '9c2cb36398b6e2ad',
    #                              'img_002.jpg': 'f684a3236b4a5377',
    #                              ...}
    encodings = phasher.encode_images(
        image_dir=image_dir,
        # num_enc_workers=0,
    )

    # Find duplicates using the generated encodings
    # duplicates: dict[str, list[str]] = {'img_001.jpg': [],
    #                                     'img_002.jpg': ['img_003.jpg'],
    #                                     'img_003.jpg': ['img_002.jpg'],
    #                                     ...}
    duplicates = phasher.find_duplicates(
        encoding_map=encodings,
        # max_distance_threshold=0,
        # num_enc_workers=0,
        # num_dist_workers=0,
    )

    # plot duplicates obtained for a given file using the duplicates dictionary
    # plot_duplicates(image_dir=image_dir,
    #                 duplicate_map=duplicates,
    #                 filename='1_17PM_0980.jpg')

    dedup_list, dedup_set = [], set()
    duplicates = dict(sorted(duplicates.items()))
    for img_name, dup_imgs in tqdm(duplicates.items()):
        if img_name in dedup_set:
            continue
        dedup_list.append(img_name)
        dedup_set.add(img_name)
        dedup_set.update(dup_imgs)
    print(f'Number of total images: {len(duplicates)}')
    print(f'Number of deduplicated images: {len(dedup_list)}')

    save_dir = image_dir.parent / f'{image_dir.stem}_phash_dedup_distance_10'
    os.umask(0)
    save_dir.mkdir(exist_ok=True)

    save_txt = save_dir / 'phash_dedup_distance_10.txt'
    with open(save_txt, 'w', encoding='utf-8') as f:
        f.writelines([f'{image_dir / img_name}\n' for img_name in dedup_list])

    if copy:
        for img_name in tqdm(dedup_list):
            shutil.copy(image_dir / img_name, save_dir)


def dedup_dirs(root):
    root = Path(root)
    img_dirs = sorted([d for d in root.glob('**/images')
                       if d.is_dir() and 'dedup' not in d.stem])
    for i, img_dir in enumerate(tqdm(img_dirs)):
        img_paths = list(img_dir.glob('*.jpg'))
        if not len(img_paths):
            continue
        dedup(img_dir)


def check_dedup(root):
    root = Path(root)
    img_dirs = sorted([d for d in root.glob('*') if d.is_dir()])
    for img_dir in tqdm(img_dirs):
        dedup_dir = img_dir.parent / f'{img_dir.stem}_phash_dedup_distance_0'
        if not dedup_dir.exists():
            continue

        total_imgs = sorted(img_dir.glob('*.jpg'))
        dedup_imgs = sorted(dedup_dir.glob('*.jpg'))

        if len(dedup_imgs) / len(total_imgs) < 1 / 10:
            print(img_dir)


def check_clusters(image_dir, copy=True, dist=1):
    image_dir = Path(image_dir)
    phasher = PHash()

    # Generate encodings for all images in an image directory
    # encodings: dict[str, str] = {'img_001.jpg': '9c2cb36398b6e2ad',
    #                              'img_002.jpg': 'f684a3236b4a5377',
    #                              ...}
    encodings = phasher.encode_images(
        image_dir=image_dir,
        # num_enc_workers=0,
    )

    # Find duplicates using the generated encodings
    # duplicates: dict[str, list[str]] = {'img_001.jpg': [],
    #                                     'img_002.jpg': ['img_003.jpg'],
    #                                     'img_003.jpg': ['img_002.jpg'],
    #                                     ...}
    duplicates = phasher.find_duplicates(
        encoding_map=encodings,
        max_distance_threshold=dist,
        # num_enc_workers=0,
        # num_dist_workers=0,
    )

    clusters, dup_img_names_set = [], set()
    for img_name, dup_img_names in tqdm(duplicates.items()):
        if img_name in dup_img_names_set:
            continue
        cluster = [(image_dir / img_name).as_posix()]
        dup_img_names_set.add(img_name)
        for dup_img_name in dup_img_names:
            if dup_img_name in dup_img_names_set:
                continue
            cluster.append((image_dir / dup_img_name).as_posix())
            dup_img_names_set.add(dup_img_name)
        clusters.append(cluster)

    cluster_dir = image_dir.parent / f'images_for_dedup_phash_dist_{dist}'
    os.umask(0)
    cluster_dir.mkdir(parents=True, exist_ok=True)

    num_clusters = len(clusters)
    zfill_width = len(str(num_clusters))

    c_to_img_paths = {f'c{str(i).zfill(zfill_width)}': c
                      for i, c in enumerate(clusters)}
    json_path = cluster_dir.parent / f'phash_dist_{dist}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(c_to_img_paths, f, indent=4)

    if copy:
        for i, cluster in enumerate(tqdm(clusters)):
            ci = str(i).zfill(zfill_width)
            for src in cluster:
                src = Path(src)
                dst = cluster_dir / f'c{ci}_{src.name}'
                shutil.copy(src, dst)


def main():
    # dedup('/home/ganhao/data/ppvpd/processed/v1.1.0_cleaning/images_for_dedup',
    #       copy=False)
    check_clusters(
        '/home/ganhao/data/ppvpd/processed/v1.1.0_cleaning/dedup',
        copy=True,
    )


if __name__ == '__main__':
    main()
