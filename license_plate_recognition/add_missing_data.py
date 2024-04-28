import csv
import numpy as np
import pandas as pd
import ast




def interpolate_and_write(csv_filepath: str, output_filepath: str):

    df = pd.read_csv(csv_filepath)
    interpolated_df = interpolate_bounding_box(df)
    interpolated_df.to_csv(output_filepath)


def interpolate_bounding_box(df:pd.DataFrame):
    # fill NaN values in car_id column
    try:
        x = len(df) - len(df['frame_num'].unique()) +1
        interpolated_df = df.set_index(df['frame_num'], drop=True).reindex(list(range(df['frame_num'].max()+x))).drop(columns=['index','frame_num'])
        for car_id in interpolated_df.car_id.unique():
            temp_df = interpolated_df[interpolated_df['car_id'] == car_id]
            if len(temp_df)>1:
                first_frame = temp_df.index[0]
                last_frame = temp_df.index[-1]
                interpolated_df.loc[first_frame:last_frame, 'car_id'] = car_id
        
    except Exception as e:
        raise e
    
    # interpolate columns that contains bounding boxes

    try :
        interpolated_df = interpolated_df.groupby('car_id').apply(interpolate_bbox_func).reset_index(drop=True)

    except BaseException as e:
        raise e

    return interpolated_df

def interpolate_bbox_func(group):
    
    for col in ['car_bbox', 'license_plate_bbox'] :  
        group[col] = group[col].dropna().apply(ast.literal_eval)
        df_bbox = pd.DataFrame({
            'x1': group[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 1 else None),
            'y1': group[col].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None),
            'x2': group[col].apply(lambda x: x[2] if isinstance(x, list) and len(x) > 1 else None),
            'y2': group[col].apply(lambda x: x[3] if isinstance(x, list) and len(x) > 1 else None)
        })
        print(df_bbox)
        df_bbox = df_bbox.interpolate('linear', axis=0)
        group[col] = df_bbox.apply(lambda row: row.tolist(), axis=1)

    group = group.fillna(method = 'ffill')
    
    return group




