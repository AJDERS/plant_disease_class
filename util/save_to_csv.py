import pandas as pd

def save_result_to_csv(csv_path, data):
    try:
        csv_df = pd.read_csv(csv_path)
        success = True
    except:
        csv_df = pd.DataFrame(data, index=[0])
        csv_df.to_csv(csv_path, index=False)
        success = False
    if success:
        next_index = len(csv_df.index)
        new_df = pd.DataFrame(data, index=[next_index-1])
        csv_df = csv_df.append(new_df, ignore_index = True)
        csv_df.to_csv(csv_path, index=False)


def record_is_in_csv(csv_path, timestamp):
    try:
        csv_df = pd.read_csv(csv_path)
        success = True
    except:
        return False
    if timestamp in csv_df['time-stamp'].values:
        return True
    else:
        return False