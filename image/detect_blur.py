from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm


"""Poor performance"""


def calc_laplacian_var(p):
    p = str(p)
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (640, 640))
    v = cv2.Laplacian(img, cv2.CV_64F).var()
    return v

    h, w = img.shape[:2]
    total, count = 0, 0
    for row in range(0, h, 32):
        for col in range(0, w, 32):
            patch = img[row:row+32, col:col+32]
            if patch.size > 0:
                total += cv2.Laplacian(patch, cv2.CV_64F).var()
                count += 1
    v = total / count if count > 0 else 0

    return v


def detect_blur(root, num_threads=8):
    root = Path(root)
    sufs = {'.jpg', '.jpeg', '.png'}
    paths = sorted([p for p in root.glob('**/*.*') if p.suffix.lower() in sufs])

    with ThreadPoolExecutor(num_threads) as executor:
        vs = list(tqdm(executor.map(calc_laplacian_var, paths),
                       total=len(paths),
                       smoothing=0))

    with open(root / 'laplacian_vars_patch.csv', 'w', encoding='utf-8') as f:
        f.write('img_path,laplacian_var\n')
        f.writelines([f'{p},{v:0>6f}\n' for p, v in zip(paths, vs)])


def main():
    detect_blur('/home/ganhao/projects/OVD_pytools/docs/example_images/blur')


if __name__ == '__main__':
    main()
