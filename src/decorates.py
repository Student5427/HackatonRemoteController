from time import time, sleep
import psutil
import os
import threading


def memory_monitor(interval, results):
    process = psutil.Process(os.getpid())
    while not results['stop']:
        memory_usage = process.memory_info().rss / 1024 ** 2  # в мегабайтах
        results['usage'].append(memory_usage)
        sleep(interval)


def time_memory(func):
    """ Декоратор измерения времени работы функции и используемой памяти """
    def wrapper(*args, **kwargs):
        print('Запуск функции:', func.__name__)

        # results = {'usage': [], 'stop': False}
        # monitor_thread = threading.Thread(target=memory_monitor, args=(0.1, results))
        # monitor_thread.start()

        start = time()
        result = func(*args, **kwargs)
        duration = time() - start

        # results['stop'] = True
        # monitor_thread.join()
        #
        # peak_memory = max(results['usage'])

        print(f'Время работы функции: {duration:.4f} секунд')
        # print(f'Пиковое использование памяти: {peak_memory:.2f} MB')
        print()

        return result

    return wrapper
