import pickle
import numpy as np
import pandas as pd

def save_result(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_result(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def load_frame_indexes_from_csv(file_path):
    # Read the CSV file using pandas
    df = pd.read_csv(file_path)

    # Check if 'Frame_Index' column exists in the CSV
    if 'Frame_Index' not in df.columns:
        raise ValueError("CSV file must contain a 'Frame_Index' column.")

    # Convert the 'Frame_Index' column to a NumPy array
    frame_indexes = np.array(df['Frame_Index'])

    return frame_indexes

def load_labels(file_path):
    """
    Load timestamps into two numpy arrays based on labels.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        tuple: Two numpy arrays for labels 1 and 2.
    """
    # Initialize empty lists for each label
    label_1 = []
    label_2 = []

    # Read the file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                ts1, ts2, label = float(parts[0]), float(parts[1]), int(parts[2])
                if label == 1:
                    label_1.append(ts1)
                elif label == 2:
                    label_2.append(ts1)

    # Convert lists to numpy arrays
    return np.array(label_1), np.array(label_2)