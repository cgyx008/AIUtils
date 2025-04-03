import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


def do_something():
    pass


def multithreads(num_threads):
    data_list = []
    with ThreadPoolExecutor(num_threads) as executor:
        list(tqdm(executor.map(do_something, data_list), total=len(data_list)))


def kill_zombie_process():
    """If GPUs don't release memory, run the command and kill related process.
    $ sudo fuser -v /dev/nvidia*
    """
    cmd_output = '''  ganhao    1020668 F...m python
                     ganhao    1020669 F...m python
                     ganhao    1020670 F...m python
                     ganhao    1020671 F...m python
                     ganhao    1020672 F...m python
                     ganhao    1020674 F...m python
                     ganhao    1020675 F...m python'''
    pids = re.findall('\d+', cmd_output)
    print(f'{pids = }')
    print(f"kill {' '.join(pids)}")


def release_gpu_memory():
    result = subprocess.run(
        ['fuser', '-v', '/dev/nvidia-uvm'],  # 命令参数列表（推荐避免 shell=True）
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # 自动解码字节为字符串（Python 3.7+）
    )
    commands = result.stderr.splitlines()[1:]
    pids = result.stdout.split()
    assert len(commands) == len(pids)

    zombie_pids = [pids[i]
                   for i in range(len(pids))
                   if 'ganhao' in commands[i] and 'nvtop' not in commands[i]]
    result = subprocess.run(
        ['kill', *zombie_pids],  # 命令参数列表（推荐避免 shell=True）
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result


if __name__ == '__main__':
    release_gpu_memory()
