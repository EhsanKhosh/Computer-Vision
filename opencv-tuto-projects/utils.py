import numpy as np
import cv2

def get_limits(color):
    """
    This function takes in a color in BGR format and returns the lower and upper HSV limits for that color.

    Args:
        color (list): A list of 3 integers representing the BGR color of the object to be detected.

    Returns:
        tuple: A tuple containing two numpy arrays, each of size 1x3, representing the lower and upper HSV limits for the given color.

    Raises:
        ValueError: If the input color is not a list of 3 integers.

    """
    if not isinstance(color, list) or len(color)!= 3:
        raise ValueError("Color must be a list of 3 integers")

    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lower_limit = hsvC[0][0][0] - 10, 100, 100
    upper_limit = hsvC[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit