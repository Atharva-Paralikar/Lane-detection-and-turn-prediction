## License
```
MIT License

Copyright (c) 2022 Atharva Paralikar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## Author
Atharva Paralikar - M.Engg Robotics Student at University of Maryland, College Park

## Overview of Lane detection

The segregation of dashed and solid lines is an important problem in Autonomous
Vehicles.

Steps followed for Segregation:

a) The region of interest is selected.

b) Thresholding is done on the region to extract the features. Canny filter is used
here as shown in the image below.

c) The canny edges are further worked on using morphological operations to get
contours of the dashed/solid lines.

d) Once the contours are obtained and filled, we calculate the number of pixels
the are present using the function cv2.countNonZero() in the left half and the
right half and compare them.

e) Since the solid line will have more pixels highlighted the segregation is easy.

f) The dashed line pixels are then colored red, and the solid ones are painted
green as required.

![Segregation of dashed and solid Line](https://github.com/Atharva-Paralikar/Lane-detection-and-turn-prediction/blob/main/docs/lane_segregation.gif)

## Turn Prediction

Steps Followed:

a) The region of interest is selected.

b) Homography is calculated for the ROI

c) Thresholding is done on the region to extract the features i.e., Yellow, and
White lines in this case

d) Homography is applied to the image to get the top view

e) The pixel coordinates of the yellow and white lines obtained are the fit into a
curve using np.polyfit()

f) The curves can be seen in the output image in real time.

g) The radius at certain points on the yellow and white curves is calculate using
the formula,
![](https://github.com/Atharva-Paralikar/Lane-detection-and-turn-prediction/blob/main/docs/radius.png)

h) The turn will on the side of the curve with lesser radius of curvature.

![Turn Prediction](https://github.com/Atharva-Paralikar/Lane-detection-and-turn-prediction/blob/main/docs/turn_predict.gif)

## Steps to Run Code

1. Copy the repository
```
git clone --recursive https://github.com/Atharva-Paralikar/Lane-detection-and-turn-prediction.git
```
2. Source the repository 
```
cd ~/Lane-detection-and-turn-prediction
```
3. Run the script 
```
python3 lane_segregation.py
```
4. Run the script 
```
python3 turn_predict.py
```

