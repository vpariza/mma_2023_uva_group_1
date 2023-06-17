

from shapely.geometry import Point
import pandas as pd

def add_points_as_coords(df:pd.DataFrame):
    df['coords'] = [Point(arr_el) for arr_el in df[['lat','lon']].to_numpy()]
    return df