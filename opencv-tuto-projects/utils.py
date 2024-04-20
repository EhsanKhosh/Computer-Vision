import numpy as np
import cv2
import webcolors

def get_limits(color: str):
    """
    This function takes in a color in BGR format and returns the lower and upper HSV limits for that color.

    Args:
        color (list): A list of 3 integers representing the BGR color of the object to be detected.

    Returns:
        tuple: A tuple containing two numpy arrays, each of size 1x3, representing the lower and upper HSV limits for the given color.

    """
           
    if color == 'red':
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([170,100,20])
        upper2 = np.array([179,255,255])
        
        return lower1, upper1, lower2, upper2
        
    else:
        hsvC = convert_name_color_to_hsv(color)
        lower_limit = hsvC[0][0][0] - 10 , 100, 100
        upper_limit = hsvC[0][0][0] + 10 , 255, 255

        lower_limit = np.array(lower_limit, dtype=np.uint8)
        upper_limit = np.array(upper_limit, dtype=np.uint8)

    
    return lower_limit, upper_limit, lower_limit, upper_limit

def convert_name_color_to_hsv(color):

    color = webcolors.name_to_rgb(color)
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)

    return hsvC


