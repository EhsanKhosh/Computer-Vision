import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(image, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # top left border
    cv2.line(image, (x1, y1), (x1, y1+line_length_y), color, thickness )
    cv2.line(image, (x1, y1), (x1 + line_length_y, y1), color, thickness )
    # bottom left border
    cv2.line(image, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(image, (x1, y2), (x1 + line_length_y, y2), color, thickness)
    # top right border
    cv2.line(image, (x2, y1), (x2-line_length_x, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1+line_length_y), color, thickness)
    # bottom right border
    cv2.line(image, (x2, y2), (x2, y2 - line_length_y), color, thickness)  
    cv2.line(image, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return image

def create_out_video(results_path: str, input_video_path: str, output_path:str):

    results = pd.read_csv(results_path)

    cap = cv2.VideoCapture(input_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    license_plate = {}

    for car_id in np.unique(results['car_id']):
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score']) 

        max_score_plates = results[(results['car_id']==car_id) & (results['license_number_score']==max_score)]
        license_plate[car_id] = {'license_crop': None,
                                'license_plate_number': max_score_plates['license_number'].iloc[0]}
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_score_plates['frame_num'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(max_score_plates['license_plate_bbox'].iloc[0].replace('[ ', '[')
                                                                                        .replace(' ', '')
                                                                                        .replace('  ', ' ')
                                                                                        .replace('   ', ' '))
        
        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1)*400 / (y2 - y1)), 400))

        license_plate[car_id]['license_crop'] = license_crop

    frame_num = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret = True

    while ret:
        ret, frame = cap.read()
        frame_num += 1

        if ret:
            temp_df = results[results['frame_num']==frame_num ]

            for idx in range(len(temp_df)):

                x1_car, y1_car, x2_car, y2_car = ast.literal_eval(temp_df.iloc[idx]['car_bbox'].replace('[ ', '[')
                                                                                            .replace(' ', '')
                                                                                            .replace("  ", ' ')
                                                                                            .replace('   ', ' '))
                draw_border(frame, (int(x1_car), int(y1_car)), (int(x2_car), int(y2_car)), thickness=25)

                x1, y1, x2, y2 = ast.literal_eval(temp_df.iloc[idx]['license_plate_bbox'].replace('[ ', '[')
                                                                                        .replace(' ', '')
                                                                                        .replace('  ', ' ')
                                                                                        .replace('   ', ' '))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                license_crop = license_plate[temp_df.iloc[idx]['car_id']]['license_crop']
                H, W, _ = license_crop.shape

                try:
                    #location of license plate
                    frame[int(y1_car) - H - 100:int(y1_car) - 100,
                        int((x2_car - W + x1_car)/2):int((x2_car + x1_car + W)/2), :] = license_crop 
                    
                    #set background of license plate
                    frame[int(y1_car) - H - 400:int(y1_car) - H - 100,
                        int((x2_car + x1_car - W)/2):int((x2_car + x1_car + W)/2), :] = (255, 255, 255)
                    
                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[temp_df.iloc[idx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17
                    )

                    cv2.putText(frame, license_plate[temp_df.iloc[idx]['car_id']]['license_plate_number'],
                                (int((x2_car + x1_car - text_width)/2) , int(y1_car - H - 250 + (text_height/2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
                    
                except:
                    pass
            out.write(frame)
            frame = cv2.resize(frame, (1280, 720)) 

    out.release()
    cap.release()   