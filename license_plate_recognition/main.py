from ultralytics import YOLO
import cv2
from sort.sort import *
from utils import *
from train_license_plate_dataset import YOLODetectorTrainer
from add_missing_data import interpolate_and_write

#initial variables 
frame_num = -1
ret = True 
# define vehicle class_ids in coco_model
vehicles = [2, 3, 5, 7] #[car, motorcycle, bus, truck]
results = {} 

#load models
mot_tracker = Sort()
coco_model = YOLO('./yolov8n.pt')
license_plate_detector = YOLO('./model/best.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

# read frames
while ret:
    ret, frame = cap.read()
    if ret:
        frame_num += 1
        results[frame_num] = {}

    # detect vehicles
        detections = coco_model(frame)[0]
        detection_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detection_.append([x1, y1, x2, y2, score])
    
        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detection_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license

            # assign license to vehicle
            x1car, y1car, x2car, y2car, car_id  =assign_plate_to_car(license, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :] 

                #process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, text_score = read_license_plate(license_plate_crop_thresh)

                # append result of new frame to results dict
                if license_plate_text is not None:
                    results[frame_num][car_id] = {'car': {'bbox': [x1car, y1car, x2car, y2car]},
                                                'license_plate':{'bbox': [ x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': text_score}
                                                    }

        # write results to csv file
        write_results_to_csv(results, './test.csv')
        interpolate_and_write('./test.csv', './test_interpolated.csv')

