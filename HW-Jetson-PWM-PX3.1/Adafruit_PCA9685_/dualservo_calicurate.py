import time
import Adafruit_PCA9685

# Set frequency to 60hz, good for servos.

# Alternatively specify a different address and/or bus:
#pwm = Adafruit_PCA9685.PCA9685(address=0x41, busnum=2)

# Configure min and max servo pulse lengths
def reset_x(angle=59):
    duty = int(float(angle) * 2.17 + 102)
    # reverse
    duty = 594 - duty
    print('x reset angle is', f' {angle}' 'duty is' f' {duty}')
    return duty #self.pwm.set_pwm(self.xaxis_pin, 0, duty)

def reset_y(angle=159):
    duty = int(float(angle) * 2.17 + 102)
    print('y reset angle is', f' {angle}' 'duty is' f' {duty}')
    return duty
        # self.pwm.set_pwm(self.yaxis_pin, 0, duty)
        #time.sleep(0.1)

def xaxis_calibrate(angle):
    duty = int(float(angle) * 2.17 + 102)
    # reverse
    duty = 594 - duty
    print('xaxis angle is', f' {angle}' 'duty is' f' {duty}')
    return duty # pwm.set_pwm(self.xaxis_pin, 0, duty)

def yaxis_calibrate(angle):
    duty = int(float(angle) * 2.17 + 102 )
    print('y-axis, angle is', f' {angle}' 'duty is' f' {duty}')
    return duty # pwm.set_pwm(self.yaxis_pin, 0, duty)
        #time.sleep(0.1)


def calibrate(opt):
    pwm = Adafruit_PCA9685.PCA9685()
    # Set frequency to 60hz, good for servos.
    pwm.set_pwm_freq(50)
    xaxis_pin = 15
    yaxis_pin = 14
    print('start cycle')
    c = 0
    while True:
        # x-min
        duty = xaxis_calibrate(opt.x_min)
        pwm.set_pwm(xaxis_pin, 0, duty)
        time.sleep(2)
        print('1st clear')
        # x-max
        duty = xaxis_calibrate(opt.x_max)
        pwm.set_pwm(xaxis_pin, 0, duty)
        time.sleep(2)
        print('2st clear')
        duty = reset_x()
        pwm.set_pwm(xaxis_pin, 0, duty)
        time.sleep(2)
        # y-min
        duty = yaxis_calibrate(opt.y_min)
        pwm.set_pwm(yaxis_pin, 0, duty)
        time.sleep(2)
        print('3st clear')
        # y-max
        duty = yaxis_calibrate(opt.y_max)
        pwm.set_pwm(yaxis_pin, 0, duty)
        time.sleep(2)
        print('4st clear')
        duty = reset_y()
        pwm.set_pwm(yaxis_pin, 0, duty)
        time.sleep(2)
        c += 1
        if c==1:
            break
