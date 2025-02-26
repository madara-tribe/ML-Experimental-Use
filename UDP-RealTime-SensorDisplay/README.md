# RealTime Display for UDP Sensor data with PyQt

Get sensor data from jetson and send to GUI on Mac (Pyside)

![sensor](https://user-images.githubusercontent.com/48679574/205445600-a379f2f5-a2ea-4c57-9166-6b7614148d82.png)


# Put into action
<img src="https://user-images.githubusercontent.com/48679574/205449053-f5c9492c-7a7e-4fe2-8b64-08f88049701c.gif" width="600" height="500"/>


# To use "matplotlib" with PyQt5

Change matplotlib backend and add to python script as follows;
```
# https://matplotlib.org/stable/users/explain/backends.html
vi ~/matplotlib/mpl-data/matplotlibrc

# add
backend: Qt5Agg
```
```python
import matplotlib
matplotlib.use('Qt5Agg')
```

## PyQt version check
```python
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print("Qt: v", QT_VERSION_STR, "\tPyQt: v", PYQT_VERSION_STR)
```


# References
- [PyQt + Matplotlib Real-Time plot Arduino](https://org-technology.com/posts/pyqt-matplotlib-realtime-plot-3.html)
