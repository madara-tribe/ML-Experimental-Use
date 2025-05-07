from enum import Enum
import numpy as np

class AngleFormula(Enum):
    ServoMax = 180
    WidthMax = 600
    HightMax = 600

def AngleX(x):
    angle = x / (AngleFormula.WidthMax.value/AngleFormula.ServoMax.value)
    return angle 

def AngleY(y):
    angle = y / (AngleFormula.HightMax.value/AngleFormula.ServoMax.value)
    return angle

def Angle2Duty(inputs, width=True):
    if width:
        deg = AngleX(inputs)
    else:
        deg = AngleY(inputs)
    print(deg)
    dc = (deg*9.5)/AngleFormula.ServoMax.value + 2.5
    return np.round(dc, decimals=2)


