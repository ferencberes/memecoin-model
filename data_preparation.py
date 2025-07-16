import pandas as pd
import numpy as np
import sys, argparse

def load_data(file_path, nrows=None):
    try:
        data = pd.read_csv(file_path, nrows=nrows)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: The file {file_path} could not be parsed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Load and process data from a CSV file.")
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    parser.add_argument('--action', choices=['buy', 'sell'], default='buy', help='Action to process (default: buy)')
    parser.add_argument('--output_dir', type=str, default='./processed', help='Path to save the processed data')
    #parser.add_argument('--filename_prefix', type=str, default='soltokens', help='Prefix for the output files')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read from the file')
    parser.add_argument('--no-features', action='store_true', help='If set, do not include features in the output')

    args = parser.parse_args()

    data = load_data(args.file_path, nrows=args.nrows*2 if args.nrows else None)
    print(dict(data.dtypes))  # Display the data types of each column

    # Filter data based on the action
    data = data[data['action'] == args.action].copy()
    if args.nrows:
        data = data.head(args.nrows)

    cols = ['token', 'user', 'timestamp', 'action']
    if not args.no_features:
        cols += ['n_txs_so_far', 'avg_price']
        if args.action == 'buy':
            cols += ['buy_period', 'avg_buy_price', 'n_buys_so_far']
        elif args.action == 'sell':
            cols += ['sell_period', 'avg_sell_price', 'n_sells_so_far']
        
    print(f"Selected columns: {cols}")

    if not args.output_dir:
        output_dir = ''
    else:
        output_dir = args.output_dir + '/'
    #output_fp = f"{output_dir}{args.filename_prefix}.csv"
    output_fp = f"{output_dir}{args.action}.csv"
    if args.no_features:
        output_fp = output_fp.replace('.csv', '_no_features.csv')
    if args.nrows:
        output_fp = output_fp.replace('.csv', f'_{args.nrows}.csv')
    selected = data[cols].copy()
    selected['action'] = np.ones(len(selected))  # Ensure 'buy' column is present
    selected['action'] = selected['action'].astype('int64')  # Convert to int64
    selected.rename(columns={'action': 'label', 'token': 'i', 'user': 'u', 'timestamp': 'ts'}, inplace=True)
    for col in selected.columns:
        if selected[col].dtype == 'object':
            selected[col] = selected[col].astype('int64')
    missing_rates = selected.isnull().mean()
    print("Missing rates per column:")
    print(missing_rates)
    print("Filling missing values with 0")
    selected.fillna(0, inplace=True)
    selected.to_csv(output_fp, index=False, header=False)

if __name__ == "__main__":
    main()
    sys.exit(0)