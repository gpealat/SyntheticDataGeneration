import argparse
import pandas as pd
import numpy as np

def create_sequences(data:np.ndarray, seq_len:int)->np.ndarray:
    """Function which takes a long array of data and split it into seq_len long sequences 

    Parameters
    ----------
    data : np.ndarray
        Long array of returns
    seq_len : int
        Length of each sequence

    Returns
    -------
    np.ndarray
        Stacked sequences of overlapping returns
    """
    # Data structure which will hold the sequence
    sequences = []

    # Looping on the input data
    for i in range(len(data) - seq_len):
        # Extracting the overlapping sequence
        seq = data[i:i+seq_len]
        # Append the list
        sequences.append(seq)

    return np.stack(sequences)  # Shape: (N, seq_len)



def create_dataset(args:argparse.Namespace):
    """Creating a dataset of path from history

    Parameters
    ----------
    args : argparse.Namespace
        Parameters of the script
    """

    # Create the filename to load
    filename = f'./data/{args.datafile}.csv'
    
    # Load the data into the data structure
    prices = pd.read_csv(filename).iloc[:,1] # Shape: (T,)

    # Add a column which represents the log returns
    log_returns =  log_returns = np.diff(np.log(prices))  # Shape: (T-1,)

    # Create sequences 
    seq_len = args.path_length  # e.g., 20 days
    sequences = create_sequences(log_returns, seq_len)
    sequences = sequences[..., np.newaxis]  # Shape: (N, seq_len, 1)

    # We sqeez one dimension down
    flatten_sequences = sequences.squeeze(-1)
    
    # To store the dataset, we put it back into a dataframe
    df = pd.DataFrame(flatten_sequences)

    # We need to change the name of the columns from int to str
    df.columns = [f'd_{seq_len-i-1}' for i in range(seq_len)]

    # We store the dataset into a parquet file with snappy compression
    output_filename = f'./data/{args.datafile}.parquet'

    # Save the data to parquet file
    df.to_parquet(output_filename, compression="snappy")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset creation arguments')
    parser.add_argument('--datafile', type=str, default='spx', help='Name of theinput dataset')
    parser.add_argument('--path_length', type=int, default=20, help='Length of the path to extract')

    # Parse the arguments passed after -m scripts.test
    args = parser.parse_args()

    create_dataset(args)