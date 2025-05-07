#!/usr/bin/env python

import Jetson.GPIO as GPIO
import time
import threading
import sys
from formura import Angle2Duty

OUTPUT_PIN1 = 32 
OUTPUT_PIN2 = 33
CYCLE = 50
t=2

def setup_device():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(OUTPUT_PIN1, GPIO.OUT, initial=GPIO.HIGH)
    pw = GPIO.PWM(OUTPUT_PIN1, CYCLE)

    GPIO.setup(OUTPUT_PIN2, GPIO.OUT, initial=GPIO.HIGH)
    ph = GPIO.PWM(OUTPUT_PIN2, CYCLE)
    return pw, ph

def pw_loop(pw):
    while flag:
        dc1 = Angle2Duty(450)
        pw.start(dc1)
        print("width dc {}".format(dc1))
        time.sleep(t)
        dc2 = Angle2Duty(500)
        pw.start(dc2)
        print("width dc {}".format(dc2))
        time.sleep(t)
        dc3 = Angle2Duty(410)
        pw.start(dc3)
        print("width dc {}".format(dc3))
        time.sleep(t)
        dc4 = Angle2Duty(300)
        pw.start(dc4)
        print("width dc {}".format(dc4))
        time.sleep(t)

def ph_loop(ph):
    while flag:
        dc1 = Angle2Duty(120)
        ph.start(dc1)
        print("ph height {}".format(dc1))
        time.sleep(t)
        dc2 = Angle2Duty(180)
        ph.start(dc2)
        print("ph height {}".format(dc2))
        time.sleep(t)
        dc3 = Angle2Duty(240)
        ph.start(dc3)
        print("ph height {}".format(dc3))
        time.sleep(t)
        dc4 = Angle2Duty(170)
        ph.start(dc4)
        print("ph height {}".format(dc4))
        time.sleep(t)
        

if __name__ == '__main__':
    flag = True
    c=0
    pw, ph = setup_device()
    th1 = threading.Thread(target=pw_loop, args=(pw,))
    th1.start()
    th2 = threading.Thread(target=ph_loop, args=(ph,))
    th2.start()
    while True:
        c +=1
        if c==3000:
            flag =False
            th1.join()
            th2.join()
            pw.stop()
            ph.stop()
            GPIO.cleanup()
            sys.exit(1)
