# robot detection

 run main.py to start

 please config line 122,123 in main.py to set the camera

## yolo model from [pfa-vision-radar](https://github.com/CarryzhangZKY/pfa_vision_radar)

If running this program in windows, please manually change the \pupil_apriltags\bindings.py

    #  line 337
    self.libc = ctypes.CDLL(str(hit))
    # to
    self.libc = ctypes.CDLL(str(hit), winmode = 0)
