from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


def do_something():
    pass


def multithreads():
    data_list = []
    with ThreadPoolExecutor(8) as executor:
        list(tqdm(executor.map(do_something, data_list), total=len(data_list)))

