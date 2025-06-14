'''
- Load new transport data collected on 24th April 2025
- Data collected using Apple Watch (Motion Tracking)
'''
import datetime
import pandas as pd
import os

def load_new_transport_data(excel_path=None, csvPath=None):
    def load_data_0424(data_index_csv):
        dfs = {}

        def read_data_for_file(filepath, datestart, dateend, device, classification):
            file_data = dfs.get(filepath)
            if file_data is None:
                file_path = os.path.join(baseDir, filepath)
                file_data = pd.read_csv(file_path)

                # Check if the file is empty
                if file_data.empty:
                    print(f"[Warning@load_new_transport_data] -> The file {file_path} is empty.")
                    return pd.DataFrame()

                ax_df = file_data[["Timestamp", "accelerationX", "accelerationY", "accelerationZ"]].copy()
                ax_df.columns = ["Timestamp", "X", "Y", "Z"]
                ax_df["Timestamp"] = ax_df["Timestamp"].map(lambda t: float(t) if isinstance(t, str) else t)
                ax_df["Timestamp"] = ax_df["Timestamp"].map(lambda t: datetime.datetime.fromtimestamp(t) if isinstance(t, (float, int)) else t)
                ax_df = ax_df.set_index(pd.DatetimeIndex(ax_df['Timestamp']))
                ax_df = ax_df.drop("Timestamp", axis=1)
                dfs[filepath] = ax_df

            datestart = datetime.datetime.fromtimestamp(datestart) # Convert string to datetime
            dateend = datetime.datetime.fromtimestamp(dateend)

            df = dfs[filepath][datestart:dateend].copy()
            df["Device"] = device
            df["Classification"] = classification

            return df

        def read_data_for_user(row):
            return read_data_for_file(row["filepath"], row["datestart"], row["dateend"], row["device"], row["classification"])

        accelerometer_data = data_index_csv.apply(read_data_for_user, axis=1)

        # Count the number of data
        data_amount_ax = datetime.timedelta(seconds = 0)
        for data in accelerometer_data:
            length = datetime.timedelta(seconds = (data.index.max() - data.index.min()).total_seconds())
            data_amount_ax += length
        print("[info@load_new_transport_data] -> Examples Axivity     " + str(data_amount_ax))

        # No need to filter the data again since it's already filtered
        # accelerometer_data = accelerometer_data.apply(
        #     lambda df: df.assign(
        #         X=butterworth_filter(df['X']),
        #         Y=butterworth_filter(df['Y']),
        #         Z=butterworth_filter(df['Z'])
        #     )
        # )

        return accelerometer_data

    # import csv data from excel file
    if excel_path or csvPath:
        if csvPath and os.path.exists(csvPath):
            csv_transport_0424 = pd.read_csv(csvPath)
            data_transport_0424 = load_data_0424(csv_transport_0424)
            return data_transport_0424
        if excel_path and not os.path.exists(csvPath):
            df = pd.read_excel(excel_path, sheet_name='Sheet1', header=0)
            baseDir = os.path.dirname(excel_path) # The same directory of excel_path
            savedPath = os.path.join(baseDir, "data_transport_index_0424.csv")
            df.to_csv(savedPath, index=False, float_format='%.4f')
            print("[info@load_new_transport_data] -> Excel file converted to CSV.")
    else:
        raise ValueError("Either excel_path or csvPath must be provided.")
    
    csv_transport_0424 = pd.read_csv(csvPath)
    data_transport_0424 = load_data_0424(csv_transport_0424)
    return data_transport_0424