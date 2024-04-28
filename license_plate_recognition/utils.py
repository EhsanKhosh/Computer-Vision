import string
import easyocr
import numpy as np
import pandas as pd

#Initialize the OCR Reader
reader = easyocr.Reader(['en'], gpu=False)

#Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'
                    }
dict_int_to_char = {'0':'O',
                    '1':'I',
                    'J':'3',
                    'A':'4',
                    'G':'6',
                    'S':'5'}



def assign_plate_to_car(license_plate, vehicle_track_ids):
    
    x1, y1, x2, y2, score, class_id = license_plate
    found = False

    for i in range(len(vehicle_track_ids)):
        x1car, y1car, x2car, y2car, car_id = vehicle_track_ids[i]
        # find out this license plate is in bbox of that vehicle 
        if x1 > x1car and y1 > y1car and x2 < x2car and y2 < y2car: 
            car_idx = i
            found = True
            break

    if found:
        return vehicle_track_ids[car_idx]

    return -1, -1, -1, -1, -1


def license_complies_format(text):

    num_0_to_9 = np.asarray(range(10)).astype(str)
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in  num_0_to_9 or text[2] in dict_char_to_int.keys()) and \
       (text[3] in num_0_to_9 or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
       
        return True
    
    else:
        return False
    

def  format_license(text):
    
    license = ''
    mapping = {0: dict_char_to_int, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int,
               4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char}
    
    for i in range(7):
        if text[i] in mapping[i].keys():
            license += mapping[i][text[i]]
        else:
            license += text[i]

    return license


def read_license_plate(license_plate_crop):
    
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        print(detection)
        text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score
        
    return None, None





import pandas as pd

def write_results_to_csv(results, out_path):
    temp_dicts = []  # List to store dictionaries for each frame and car
    idx = 0
    for frame_num in results.keys():
        for car_id in results[frame_num].keys():
            if 'car' in results[frame_num][car_id].keys() and \
               'license_plate' in results[frame_num][car_id].keys() and \
               'text' in results[frame_num][car_id]['license_plate'].keys():
                
                temp_dict = {}  # Create a new dictionary for each frame and car
                temp_dict['frame_num'] = frame_num
                temp_dict['car_id'] = car_id
                temp_dict['car_bbox'] = results[frame_num][car_id]['car']['bbox']
                temp_dict['license_plate_bbox'] = results[frame_num][car_id]['license_plate']['bbox']
                temp_dict['license_plate_bbox_score'] = results[frame_num][car_id]['license_plate']['bbox_score']
                temp_dict['license_number'] = results[frame_num][car_id]['license_plate']['text']
                temp_dict['license_number_score'] = results[frame_num][car_id]['license_plate']['text_score']
                temp_dicts.append(temp_dict)  # Append the dictionary to the list
                idx += 1
   

    print(temp_dicts)
    df = pd.DataFrame(temp_dicts,index=range(len(temp_dicts)))  # Create DataFrame from the list of dictionaries
    df.to_csv(out_path, index=False)  # Write DataFrame to CSV without index column



