from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def imshow_feature_map(feature_map, save_path='', title=''):
    """
    Imshow and save feature map
    Args:
        feature_map: np.ndarray shape(h, w)
        save_path (str | Path): save path
        title (str): title
    """
    fig, ax = plt.subplots()

    cax = ax.imshow(feature_map, cmap='viridis')
    fig.colorbar(cax)

    ax.set_title(title)
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def visualize_feature_map():
    root = Path('/home/ganhao/data/reolink/test/20241115/ovd_000_person_man_debug_input_yuv1_no_detections9/feature_map/person_man')
    np_path = root / 'y3.npy'
    feature_maps = np.load(np_path)

    for i, feature_map in enumerate(tqdm(feature_maps.squeeze())):
        feature_map: np.ndarray

        feature_map_mean = feature_map.mean(axis=0)
        save_path = root / f'mean_and_max/{np_path.stem}_{i}_mean.png'
        imshow_feature_map(feature_map_mean, save_path, 'Y')

        feature_map_max = feature_map.max(axis=0)
        save_path = root / f'mean_and_max/{np_path.stem}_{i}_max.png'
        imshow_feature_map(feature_map_max, save_path, 'Y')



def main():
    visualize_feature_map()


if __name__ == '__main__':
    main()
