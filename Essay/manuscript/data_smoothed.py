import pandas as pd
import numpy as np
from typing import List, Tuple


def smooth(csv_path, weight=0.99) -> Tuple:
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int, 'Value': np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    # save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    # save.to_csv('smooth_' + "{}.csv".format("PSNR"))
    step_list = data['Step'].values.tolist()
    return step_list, smoothed


if __name__ == '__main__':
    smooth('test.csv')
