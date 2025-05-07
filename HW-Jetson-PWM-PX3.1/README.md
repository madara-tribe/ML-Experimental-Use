# Jetson Nano PWM as HW calibrater
# abstract

Jetson's PWM(Pulse Width Modulation) to move 2 Servomortor with multi-thread.

```sh
- JetPack > 4.6
- python3 3.6.9
- opencv 4.1.2
- Servo mortor :  TowerPro SG-90, 
```

<b>Jetson Nano and 2 Serbo mortor</b>

<img src="https://user-images.githubusercontent.com/48679574/211758179-abc3911f-4c45-4318-9de4-2c95f379bbd9.jpg" width="350" height="200"/>

# Adafruit-PCA9685 (PWM controller)

It can control multi-servo motor (maximum is 16). and script don't need multithread process.

<img src="https://user-images.githubusercontent.com/48679574/216533582-243f2e2c-19d7-4bae-bfee-6245f2e5709d.jpg" width="350" height="200"/>


# Setup

```zsh
# rewrite user name and install GPIO
sudo bash setup/install_gpio.sh
# setup pins for servo PWM
sudo bash setup/setpwm.sh
# Adafruit-PCA9685
pip3 install Adafruit_PCA9685
# i2cdetect -y -r 1
# rewrite I2C.py file
vi ~/.local/lib/python3.6/site-packages/Adafruit_GPIO/I2C.py
```

# Calibrator

<b>GUI calibrate</b>
```zsh
$ python3 calibrater.py -g
```

<b>automatic calibrate</b>
```zsh
$ python3 calibrater.py -c
```

# Similater

## x-axis
<b>Simulation cycle</b> : <b>reset to default point</b> => <b>move from x-min to x-max</b> => <b>reset to default point</b>
```zsh
$ python3 simulator.py -a x
```
<b>YOUTUBE MOVIE LINK</b>

[![PX3.1 X-AXIS](https://github.com/madara-tribe/HW-Jetson-PWM-PX3.1/assets/48679574/011ac097-25a8-454a-840d-e22052cb17a1)](https://youtube.com/shorts/qoQlibyCR1c?si=ihqIfNyfjMD7pwjR)


## y-axis 
<b>Simulation cycle</b> : <b>reset to default point</b> => <b>move from y-min to y-max</b> => <b>reset to default point</b>
```zsh
$ python3 simulator.py -a y
```
<b>YOUTUBE MOVIE LINK</b>

[![PX3.1 Y-AXIS](https://github.com/madara-tribe/HW-Jetson-PWM-PX3.1/assets/48679574/3785e794-45bc-49a2-9b1b-ca8ba514d7c6)](https://youtu.be/EXGpjnlBxCU?si=efqPf6jmlTXRQoDE)



# References
- [jetson-gpio](https://github.com/NVIDIA/jetson-gpio/tree/master/samples)
- [Jetson Nano の 2 つのハードウェア PWM を使用してみた](https://qiita.com/kitazaki/items/2c9deb912f11106d1215)
- [jetson_nano_grove](https://github.com/kitazaki/jetson_nano_grove)
