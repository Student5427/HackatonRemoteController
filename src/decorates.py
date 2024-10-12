from time import time, sleep
from typing import Callable, Any
import psutil
import os
import threading


def memory_monitor(interval, results) -> None:
    """
    Функция измерения использованной памяти

    :param interval: Период измерений объема памяти
    :param results: Словарь для хранения результатов мониторинга памяти
    :return:
    """
    process = psutil.Process(os.getpid())
    while not results['stop']:
        memory_usage = process.memory_info().rss / 1024 ** 2  # в мегабайтах
        results['usage'].append(memory_usage)
        sleep(interval)


def time_memory(func: Callable) -> Callable:
    """ Декоратор измерения времени работы функции и используемой памяти """
    def wrapper(*args, **kwargs) -> Any:
        results = {'usage': [], 'stop': False}
        monitor_thread = threading.Thread(target=memory_monitor, args=(0.1, results))
        monitor_thread.start()

        start = time()
        result = func(*args, **kwargs)
        duration = time() - start

        results['stop'] = True
        monitor_thread.join()

        peak_memory = max(results['usage'])

        print(f'{func.__name__} - время работы: {duration:.4f} секунд')
        print(f'{func.__name__} - пиковое использование памяти: {peak_memory:.2f} MB')
        print()

        return result

    return wrapper
