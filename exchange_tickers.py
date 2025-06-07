'''
Exchange ticker lists for stock screening application.
This file contains ticker lists for multiple exchanges: IDX, NYSE, NASDAQ, and AMEX.
'''

import pandas as pd
import os

# Import the IDX tickers from the original file
from idx_all_tickers import IDX_ALL_TICKERS_YF

# Function to load tickers from CSV files
def load_tickers_from_csv(csv_path):
    """Load tickers from CSV file and return as a list."""
    try:
        df = pd.read_csv(csv_path)
        # Extract just the Symbol column and convert to list
        tickers = df['Symbol'].tolist()
        # Filter out any non-string values or empty strings
        tickers = [str(ticker) for ticker in tickers if ticker and isinstance(ticker, (str, int, float))]
        return tickers
    except Exception as e:
        print(f"Error loading tickers from {csv_path}: {e}")
        return []

# Define paths to CSV files
NYSE_CSV_PATH = '/home/ubuntu/upload/NYSEtickers.csv'
NASDAQ_CSV_PATH = '/home/ubuntu/upload/NASDAQtickers.csv'
AMEX_CSV_PATH = '/home/ubuntu/upload/AMEXtickers.csv'

# Load tickers from CSV files
NYSE_TICKERS = load_tickers_from_csv(NYSE_CSV_PATH)
NASDAQ_TICKERS = load_tickers_from_csv(NASDAQ_CSV_PATH)
AMEX_TICKERS = load_tickers_from_csv(AMEX_CSV_PATH)

# Create a dictionary mapping exchange names to ticker lists
EXCHANGE_TICKERS = {
    'IDX': IDX_ALL_TICKERS_YF,
    'NYSE': NYSE_TICKERS,
    'NASDAQ': NASDAQ_TICKERS,
    'AMEX': AMEX_TICKERS
}

# Function to get tickers for a specific exchange
def get_exchange_tickers(exchange_name):
    """Get the list of tickers for the specified exchange."""
    return EXCHANGE_TICKERS.get(exchange_name, [])

# Function to get exchange information (name and ticker count)
def get_exchange_info():
    """Get information about available exchanges."""
    return {
        'IDX': {'name': 'Indonesia Stock Exchange', 'count': len(IDX_ALL_TICKERS_YF)},
        'NYSE': {'name': 'New York Stock Exchange', 'count': len(NYSE_TICKERS)},
        'NASDAQ': {'name': 'NASDAQ', 'count': len(NASDAQ_TICKERS)},
        'AMEX': {'name': 'American Stock Exchange', 'count': len(AMEX_TICKERS)}
    }

# Print ticker counts for debugging
if __name__ == "__main__":
    print(f"IDX Tickers: {len(IDX_ALL_TICKERS_YF)}")
    print(f"NYSE Tickers: {len(NYSE_TICKERS)}")
    print(f"NASDAQ Tickers: {len(NASDAQ_TICKERS)}")
    print(f"AMEX Tickers: {len(AMEX_TICKERS)}")
