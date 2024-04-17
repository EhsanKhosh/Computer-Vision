import cv2
from utils import get_limits
from PIL import Image

cap = cv2.VideoCapture(0)

color = 'blue'

while True:
    ret, frame = cap.read()

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hue1, upper_hue1, lower_hue2, upper_hue2  = get_limits(color)
    lower_mask = cv2.inRange(hsv_image, lower_hue1, upper_hue1)
    upper_mask = cv2.inRange(hsv_image, lower_hue2, upper_hue2)
    
    full_mask = lower_mask + upper_mask

    mask_ = Image.fromarray(full_mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x, y, w, h = bbox
        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

