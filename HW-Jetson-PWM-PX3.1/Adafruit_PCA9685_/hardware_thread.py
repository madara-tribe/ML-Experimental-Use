import sys
from multiprocessing import Queue
import time
import Adafruit_PCA9685

xaxis_pin = 15
yaxis_pin = 14

def reset_x(angle=59):
    duty = int(float(angle) * 2.17 + 102)
    # reverse
    duty = 594 - duty
    print('x reset angle is', f' {angle}' 'duty is' f' {duty}')
    return duty

def reset_y(angle=159):
    duty = int(float(angle) * 2.17 + 102)
    print('y reset angle is', f' {angle}' 'duty is' f' {duty}')
    return duty


def hw_thread(q:Queue, opt):
    pwm = Adafruit_PCA9685.PCA9685()
    # Set frequency to 60hz, good for servos.
    pwm.set_pwm_freq(50)
    duty = reset_x(angle=opt.x_defalut)
    pwm.set_pwm(xaxis_pin, 0, duty)
    time.sleep(0.5)
    duty = reset_y(angle=opt.y_defalut)
    pwm.set_pwm(yaxis_pin, 0, duty)
    time.sleep(0.5)
    stack = []
    while True:
        duty = q.get()
        axis, duty, angle = duty
        stack.append(duty)
        print('axis is {}'.format(axis), 'min is {0}, max is {1}'.format(int(min(stack)), int(max(stack))))
        if axis=='x':
            pwm.set_pwm(xaxis_pin, 0, duty)
            if angle==int(opt.x_max-1):
                duty = reset_x(angle=opt.x_defalut)
                pwm.set_pwm(xaxis_pin, 0, duty)
                time.sleep(1)
                break
        elif axis=='y':
            pwm.set_pwm(yaxis_pin, 0, duty)
            if angle==int(opt.y_max-1):
                duty = reset_y(angle=opt.y_defalut)
                pwm.set_pwm(yaxis_pin, 0, duty)
                time.sleep(1)
                break
    sys.exit()

if __name__ == "__main__":
    q = Queue()
    hw_thread(q, opt=None)
    
