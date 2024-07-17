import time

import GPUtil
from loguru import logger


def log_gpu_temp():
    logger.remove()  # Remove default logger, don't want to see logs in console
    logger.add('gpu_temp.log', retention='7 days',
               format='{time:YYYY-MM-DD HH:mm:ss} {message}')

    while True:
        temps = [gpu.temperature for gpu in GPUtil.getGPUs()]
        logger.info(temps)
        time.sleep(1)


def main():
    log_gpu_temp()


if __name__ == '__main__':
    main()
