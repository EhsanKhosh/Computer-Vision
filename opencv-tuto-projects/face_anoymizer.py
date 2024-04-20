import cv2
import mediapipe as mp
import argparse
import os



output_dir =  './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_img_face_detection(img, face_detection):

    
    out = face_detection.process(img)
    H, W, _ = img.shape

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1* W)
            y1 = int(y1* H)
            w = int(w* W)
            h = int(h* H)

            img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 255), 2)
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img

args = argparse.ArgumentParser()

args.add_argument('--mode', default='image')
args.add_argument('--filepath', default='./image.jpg')

args = args.parse_args()

mp_face_detection = mp.solutions.face_detection


with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:

    if args.mode in ["image"]:

        img = cv2.imread(args.filepath)
        img  = process_img_face_detection(img, face_detection)
        cv2.imwrite(os.path.join(output_dir, 'output_face_anoymizer.jpg'), img)

    elif args.mode in ['video']:
        
        cap = cv2.VideoCapture(args.filepath)
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:

            frame = process_img_face_detection(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

            
        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        while True:

            frame = process_img_face_detection(frame, face_detection)
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
            
        cap.release()
            







