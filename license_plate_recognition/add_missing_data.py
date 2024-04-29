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
        interpolated_df = df.copy()
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
        df_bbox = df_bbox.interpolate('linear', axis=0)
        group[col] = df_bbox.apply(lambda row: row.tolist(), axis=1)

    group = group.fillna(method = 'ffill')
    
    return group




