import cv2
from utils import get_limits

cap = cv2.VideoCapture(0)

color = 'red'

while True:
    ret, frame = cap.read()

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hue1, upper_hue1, lower_hue2, upper_hue2  = get_limits(color)
    lower_mask = cv2.inRange(hsv_image, lower_hue1, upper_hue1)
    upper_mask = cv2.inRange(hsv_image, lower_hue2, upper_hue2)
    
    full_mask = lower_mask + upper_mask


    cv2.imshow('frame', full_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

