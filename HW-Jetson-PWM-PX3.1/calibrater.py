import argparse
import os
from tkinter import Tk


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', action='store_true', help='calibrate gui mode')
    parser.add_argument('-c', '--command', action='store_true', help='calibrate command line mode')
    parser.add_argument('--x_max', type=int, default=180, help='x angle max')
    parser.add_argument('--x_min', type=int, default=0, help='x angle min')
    parser.add_argument('--y_max', type=int, default=180, help='y angle max')
    parser.add_argument('--y_min', type=int, default=0, help='y angle min')
    parser.add_argument('--y_defalut', type=int, default=169, help='y default angle')
    parser.add_argument('--x_defalut', type=int, default=59, help='x default angle')
    opt = parser.parse_args()
    return opt


def main(opt):
    if opt.gui:
        from Adafruit_PCA9685_.tkinter_pan_tilt import App
        root = Tk()
        root.wm_title('Servo Control')
        app = App(root)
        root.geometry("220x120+0+0")
        root.mainloop()
    elif opt.command:
        from Adafruit_PCA9685_.dualservo_calicurate import calibrate
        calibrate(opt)


if __name__=='__main__':
    opt = get_parser()
    main(opt)
