import argparse
from multiprocessing import Process, Queue

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--axis', type=str, default='x', help='specified axis [x] or [y]')
    parser.add_argument('--x_max', type=int, default=90, help='x angle max')
    parser.add_argument('--x_min', type=int, default=40, help='x angle min')
    parser.add_argument('--y_max', type=int, default=180, help='y angle max')
    parser.add_argument('--y_min', type=int, default=140, help='y angle min')
    parser.add_argument('--y_defalut', type=int, default=169, help='y default angle')
    parser.add_argument('--x_defalut', type=int, default=59, help='x default angle')
    parser.add_argument('--timer', type=int, default=0.2, help='interval time')
    opt = parser.parse_args()
    return opt
    
    
def software_threads(q):
    opt = get_parser()
    from Adafruit_PCA9685_.software_thread import sw_thread
    sw_thread(q, opt=opt)
    
def hardware_threads(q):
    opt = get_parser()
    from Adafruit_PCA9685_.hardware_thread import hw_thread
    hw_thread(q, opt=opt)
    
if __name__ == '__main__':
    q = Queue()
    p1 = Process(target = software_threads, args=(q,))
    p2 = Process(target = hardware_threads, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
