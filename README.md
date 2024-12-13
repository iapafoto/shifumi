# About
Shufumi (Stone Paper Scissors) ultimate warrior implemented using mediapipe handtracking and a glsl shader
Shufumi (Stone Paper Scissors) ultimate warrior implemented using [mediapipe handtracking](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and a [glsl shader](https://www.shadertoy.com/view/4cGBWm)

# Configuration
mediapipe requires python version 3.9 - 3.12 (64-bit)
```
$ python -m pip install msvc-runtime
$ python -m pip install mediapipe
```

# Start
Version using mediapipe.solutions.hands (best detection)
```
$ python shifumi.py
```

Version using mediapipe.tasks.python.vision.HandLandmarker (up to date way to use mediapipe)
```
$ python shifumi2.py
```

# Video
click the picture to open the short video demonstration 
[![Image2](https://github.com/iapafoto/shifumi/blob/main/demo/demo.png)](https://www.youtube.com/watch?v=koMgq2-sAQ0)

# License
Licensed under the Apache License, Version 2.0. See LICENSE.txt.
