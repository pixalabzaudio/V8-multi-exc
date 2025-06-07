import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import traceback
import concurrent.futures
import threading
from functools import partial
import matplotlib.pyplot as plt
import io
import base64

# Import the exchange ticker lists
from exchange_tickers import get_exchange_tickers, get_exchange_info

# Constants
MAX_TICKERS = 950
# Relaxed Default Fundamental Filters
DEFAULT_MIN_NI = 0.1  # Default minimum Net Income in trillion IDR (Relaxed from 1.0)
DEFAULT_MAX_PE = 30.0  # Default maximum P/E ratio (Relaxed from 15.0)
DEFAULT_MAX_PB = 2.5  # Default maximum P/B ratio (Relaxed from 1.5)
DEFAULT_MIN_GROWTH = -20.0 # Default minimum YoY growth (Relaxed from 0.0)

RSI_PERIOD = 25  # Period for RSI calculation
OVERSOLD_THRESHOLD = 30
OVERBOUGHT_THRESHOLD = 70
MAX_WORKERS = 10
BATCH_SIZE = 50

# --- Helper function for Wilder's RSI ---
def calculate_rsi_wilder(prices, period=RSI_PERIOD, ticker="N/A"):
    '''Calculate RSI using Wilder's smoothing method.'''
    print(f"[{ticker}] Calculating RSI Wilder: Input prices length = {len(prices)}")
    delta = prices.diff()
    delta = delta[1:]
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Check if enough data for initial SMA
    if len(gain) < period:
        print(f"[{ticker}] RSI Wilder Error: Not enough gain/loss data (need {period}, got {len(gain)}).")
        return pd.Series(dtype=float)

    # Calculate initial average gain and loss using SMA
    try:
        avg_gain_series = gain.rolling(window=period, min_periods=period).mean()
        avg_loss_series = loss.rolling(window=period, min_periods=period).mean()
        
        first_valid_index = period - 1
        while first_valid_index < len(avg_gain_series) and pd.isna(avg_gain_series.iloc[first_valid_index]):
            first_valid_index += 1
            
        if first_valid_index >= len(avg_gain_series):
             print(f"[{ticker}] RSI Wilder Error: Not enough valid data points after rolling SMA.")
             return pd.Series(dtype=float)
             
        avg_gain = avg_gain_series.iloc[first_valid_index]
        avg_loss = avg_loss_series.iloc[first_valid_index]
        
        if pd.isna(avg_gain) or pd.isna(avg_loss):
             print(f"[{ticker}] RSI Wilder Error: Initial SMA calculation resulted in NaN (AvgGain: {avg_gain}, AvgLoss: {avg_loss}).")
             return pd.Series(dtype=float)

        print(f"[{ticker}] RSI Wilder Initial AvgGain: {avg_gain:.4f}, AvgLoss: {avg_loss:.4f}")

    except Exception as e:
        print(f"[{ticker}] RSI Wilder Error during initial SMA: {e}")
        return pd.Series(dtype=float)

    # Initialize arrays for Wilder's averages
    wilder_avg_gain = np.array([avg_gain])
    wilder_avg_loss = np.array([avg_loss])

    # Calculate subsequent averages using Wilder's smoothing
    start_calc_index = first_valid_index + 1
    for i in range(start_calc_index, len(gain)):
        current_gain = gain.iloc[i] if not pd.isna(gain.iloc[i]) else 0
        current_loss = loss.iloc[i] if not pd.isna(loss.iloc[i]) else 0
        
        avg_gain = (wilder_avg_gain[-1] * (period - 1) + current_gain) / period
        avg_loss = (wilder_avg_loss[-1] * (period - 1) + current_loss) / period
        wilder_avg_gain = np.append(wilder_avg_gain, avg_gain)
        wilder_avg_loss = np.append(wilder_avg_loss, avg_loss)

    # Handle division by zero for avg_loss
    rs = np.divide(wilder_avg_gain, wilder_avg_loss, out=np.full_like(wilder_avg_gain, np.inf), where=wilder_avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Return the full RSI series aligned with the original price index
    rsi_index = prices.index[start_calc_index + 1 : start_calc_index + 1 + len(rsi)]
    if len(rsi) != len(rsi_index):
         print(f"[{ticker}] RSI Wilder Warning: Index alignment mismatch (RSI len {len(rsi)}, Index len {len(rsi_index)}). Returning values only.")
         rsi_series = pd.Series(rsi)
    else:
        rsi_series = pd.Series(rsi, index=rsi_index)
        
    print(f"[{ticker}] RSI Wilder Calculation successful. Output series length: {len(rsi_series)}")
    return rsi_series


# Cache technical data for 5 minutes (300 seconds)
@st.cache_data(ttl=300)
def get_rsi_yfinance(ticker):
    '''
    Calculate RSI for a given ticker using Wilder's smoothing (yfinance version with enhanced logging).
    Returns: (rsi_value, signal, rsi_history) or None if data unavailable or calculation fails
    '''
    print(f"[{ticker}] --- Starting get_rsi_yfinance --- ")
    try:
        print(f"[{ticker}] Initializing yf.Ticker...")
        stock = yf.Ticker(ticker)
        if not stock:
             print(f"[{ticker}] yf.Ticker initialization failed.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: yf.Ticker initialization failed."
             return None
             
        # Fetch enough history for RSI calculation
        fetch_period = "6mo"
        fetch_interval = "1d"
        print(f"[{ticker}] Fetching history: period='{fetch_period}', interval='{fetch_interval}'...")
        hist = stock.history(period=fetch_period, interval=fetch_interval)
        print(f"[{ticker}] History fetched. Shape: {hist.shape}")

        if hist.empty:
            print(f"[{ticker}] RSI Error: Fetched history is empty.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Fetched history is empty."
            return None
            
        if "Close" not in hist.columns:
            print(f"[{ticker}] RSI Error: 'Close' column not found in fetched history.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: 'Close' column not found in history."
            return None
            
        close_prices = hist["Close"].dropna() # Drop NaNs from close prices
        print(f"[{ticker}] Close prices extracted. Length after dropna: {len(close_prices)}")

        if len(close_prices) < RSI_PERIOD + 1:
            print(f"[{ticker}] RSI Error: Not enough valid historical data points (need {RSI_PERIOD + 1}, got {len(close_prices)}).")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Not enough valid historical data (need {RSI_PERIOD + 1}, got {len(close_prices)})."
            return None

        # Calculate RSI using Wilder's method
        print(f"[{ticker}] Calling calculate_rsi_wilder...")
        rsi_series = calculate_rsi_wilder(close_prices, period=RSI_PERIOD, ticker=ticker)

        if rsi_series.empty or rsi_series.isna().all():
            print(f"[{ticker}] RSI Error: Wilder calculation resulted in empty or all-NaN series.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Wilder calculation resulted in empty or NaN series."
            return None

        # Get the latest RSI value
        try:
            latest_rsi = rsi_series.iloc[-1]
            print(f"[{ticker}] Latest RSI value extracted: {latest_rsi:.2f}")
        except IndexError:
            print(f"[{ticker}] RSI Error: Could not get latest RSI value from series (IndexError). Series length: {len(rsi_series)}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Could not get latest RSI value (IndexError)."
            return None

        # Check if latest RSI is valid
        if pd.isna(latest_rsi):
             print(f"[{ticker}] RSI Error: Latest RSI value is NaN.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: Latest RSI value is NaN."
             return None

        # Determine signal based on RSI value
        if latest_rsi < OVERSOLD_THRESHOLD:
            signal = "Oversold"
        elif latest_rsi > OVERBOUGHT_THRESHOLD:
            signal = "Overbought"
        else:
            signal = "Neutral"
        print(f"[{ticker}] RSI Signal determined: {signal}")

        # Return latest RSI, signal, and the last RSI_PERIOD values for the chart
        rsi_history = rsi_series.dropna().tail(RSI_PERIOD).values
        if len(rsi_history) == 0:
             print(f"[{ticker}] RSI Error: No valid RSI history values found for chart after dropna/tail.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: No valid RSI history values found for chart."
             return None # Cannot create chart without history
        print(f"[{ticker}] RSI History extracted for chart. Length: {len(rsi_history)}")

        print(f"[{ticker}] --- Successfully completed get_rsi_yfinance --- ")
        return (latest_rsi, signal, rsi_history)

    except Exception as e:
        error_msg = f"RSI yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_rsi_yfinance: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None

# Cache fundamentals data for 24 hours (86400 seconds) - KEEP USING YFINANCE
@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    '''
    Retrieve fundamental financial data for a given ticker using yfinance.
    Returns: (net_income, prev_net_income, pe_ratio, pb_ratio) or None if essential data unavailable/invalid
    '''
    print(f"[{ticker}] --- Starting get_fundamentals --- ")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        print(f"[{ticker}] yfinance info and financials fetched.")

        # Initialize metrics
        net_income, prev_net_income, pe_ratio, pb_ratio = None, 0, None, None # Default prev_ni to 0

        # --- Net Income ---
        if not financials.empty and "Net Income" in financials.index:
            ni_series = financials.loc["Net Income"]
            if not ni_series.empty:
                try:
                    # Handle potential MultiIndex or different structures
                    if isinstance(ni_series, pd.Series):
                        net_income = ni_series.iloc[0] / 1e12 # Current NI in Trillion IDR
                        if len(ni_series) > 1:
                            prev_net_income = ni_series.iloc[1] / 1e12 # Previous NI in Trillion IDR
                        else:
                            print(f"[{ticker}] Fund. Warning: Previous Net Income missing (Series format). Growth may be inaccurate.")
                            st.session_state.setdefault("warnings", {})
                            st.session_state.warnings[ticker] = f"Fund. Warning: Previous Net Income missing, growth calculation may be inaccurate."
                    else: # Handle DataFrame case if structure changes
                         net_income = ni_series.iloc[0, 0] / 1e12
                         if ni_series.shape[1] > 1:
                             prev_net_income = ni_series.iloc[0, 1] / 1e12
                         else:
                             print(f"[{ticker}] Fund. Warning: Previous Net Income missing (DataFrame format). Growth may be inaccurate.")
                             st.session_state.setdefault("warnings", {})
                             st.session_state.warnings[ticker] = f"Fund. Warning: Previous Net Income missing (DataFrame format)."
                    print(f"[{ticker}] Net Income extracted: Current={net_income:.3f}T, Previous={prev_net_income:.3f}T")

                except (IndexError, TypeError, ValueError, KeyError) as e:
                     error_msg = f"Fund. Error extracting Net Income: {e}"
                     print(f"[{ticker}] {error_msg}")
                     st.session_state.setdefault("errors", {})
                     st.session_state.errors[ticker] = error_msg
                     net_income = None # Mark as invalid if extraction failed
            else:
                print(f"[{ticker}] Fund. Warning: 'Net Income' series is empty in financials.")
                st.session_state.setdefault("warnings", {})
                st.session_state.warnings[ticker] = f"Fund. Warning: 'Net Income' series is empty in financials."
        else:
            print(f"[{ticker}] Fund. Warning: 'Net Income' not found or financials empty.")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = f"Fund. Warning: 'Net Income' not found or empty in yfinance financials."

        # --- P/E Ratio ---
        pe_ratio = info.get("trailingPE", None)
        if pe_ratio is None or not isinstance(pe_ratio, (int, float)) or np.isnan(pe_ratio):
            warning_msg = f"Fund. Warning: Trailing P/E missing or invalid ({pe_ratio}). Will not filter by P/E."
            print(f"[{ticker}] {warning_msg}")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = warning_msg
            pe_ratio = None # Ensure it's None if invalid
        else:
            print(f"[{ticker}] P/E Ratio extracted: {pe_ratio:.2f}")

        # --- P/B Ratio ---
        pb_ratio = info.get("priceToBook", None)
        if pb_ratio is None or not isinstance(pb_ratio, (int, float)) or np.isnan(pb_ratio):
            warning_msg = f"Fund. Warning: P/B ratio missing or invalid ({pb_ratio}). Will not filter by P/B."
            print(f"[{ticker}] {warning_msg}")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = warning_msg
            pb_ratio = None # Ensure it's None if invalid
        else:
             print(f"[{ticker}] P/B Ratio extracted: {pb_ratio:.2f}")

        # --- Check Essential Data for Filtering ---
        if net_income is None or not isinstance(net_income, (int, float)) or np.isnan(net_income):
            error_msg = f"Fund. Error: Net Income is missing or invalid ({net_income}). Cannot apply fundamental filters."
            print(f"[{ticker}] {error_msg}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = error_msg
            return None # Cannot proceed without Net Income

        # Return collected data (prev_net_income defaults to 0 if missing)
        print(f"[{ticker}] --- Successfully completed get_fundamentals --- ")
        return (net_income, prev_net_income, pe_ratio, pb_ratio)

    except Exception as e:
        error_msg = f"Fund. yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_fundamentals: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None


def process_ticker_technical_first(ticker, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral, exchange):
    '''
    Process a single ticker with technical filters first (yfinance version).
    Returns: [ticker_symbol, rsi, signal, rsi_history] or None if not matching criteria
    '''
    print(f"[{ticker}] Processing technical filters...")
    try:
        # Use the yfinance-based function with enhanced logging
        rsi_data = get_rsi_yfinance(ticker)
        if not rsi_data:
            # Error/Warning logged in get_rsi_yfinance
            print(f"[{ticker}] Skipping technical processing due to RSI fetch/calc failure.")
            return None

        rsi, signal, rsi_history = rsi_data

        # Apply RSI range filter
        if (rsi_min > 0 and rsi < rsi_min) or (rsi_max < 100 and rsi > rsi_max):
            print(f"[{ticker}] Filtering out: RSI {rsi:.1f} outside range {rsi_min}-{rsi_max}.")
            st.session_state.setdefault("filtered_out_technical", {})
            st.session_state.filtered_out_technical[ticker] = f"Filtered out: RSI {rsi:.1f} outside range {rsi_min}-{rsi_max}."
            return None

        # Apply RSI signal filters
        if (signal == "Oversold" and not show_oversold) or \
           (signal == "Overbought" and not show_overbought) or \
           (signal == "Neutral" and not show_neutral):
            print(f"[{ticker}] Filtering out: Signal '{signal}' not selected.")
            st.session_state.setdefault("filtered_out_technical", {})
            st.session_state.filtered_out_technical[ticker] = f"Filtered out: Signal '{signal}' not selected."
            return None

        # For IDX tickers, remove the .JK suffix for display
        if exchange == 'IDX':
            ticker_symbol = ticker.replace(".JK", "")
        else:
            ticker_symbol = ticker
            
        print(f"[{ticker}] Passed technical filters: {ticker_symbol} (RSI: {rsi:.1f}, Signal: {signal})")
        return [ticker_symbol, rsi, signal, rsi_history]

    except Exception as e:
        error_msg = f"Tech. Process Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in process_ticker_technical_first: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None

def apply_fundamental_filters(technical_results, min_ni, max_pe, max_pb, min_growth, exchange):
    '''
    Apply fundamental filters to stocks that passed technical screening.
    Returns: List of stocks with both technical and fundamental data
    '''
    final_results = []
    print(f"Applying fundamental filters to {len(technical_results)} stocks...")

    # Use threading for fundamental data fetching (as it uses yfinance)
    fund_results = {}
    def fetch_fund(ticker_symbol):
        # For IDX tickers, add the .JK suffix for yfinance
        if exchange == 'IDX':
            ticker = f"{ticker_symbol}.JK"
        else:
            ticker = ticker_symbol
            
        print(f"[{ticker}] Submitting fundamental fetch job...")
        fund_data = get_fundamentals(ticker)
        if fund_data:
            fund_results[ticker_symbol] = fund_data
            print(f"[{ticker}] Fundamental data fetch successful.")
        else:
            print(f"[{ticker}] Fundamental data fetch failed or returned None.")
            # Error logged in get_fundamentals

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_fund, result[0]) for result in technical_results]
        # Wait for all fundamental fetches to complete (no timeout)
        print(f"Waiting for {len(futures)} fundamental fetch jobs to complete...")
        concurrent.futures.wait(futures) # Removed timeout to ensure all complete
        print(f"Fundamental fetch jobs complete.")

    print(f"Fundamental data available for {len(fund_results)} stocks after fetch.")

    for result in technical_results:
        ticker_symbol, rsi, signal, rsi_history = result
        
        # For IDX tickers, add the .JK suffix for display in logs
        if exchange == 'IDX':
            ticker = f"{ticker_symbol}.JK"
        else:
            ticker = ticker_symbol

        if ticker_symbol not in fund_results:
            print(f"[{ticker}] Skipping fundamental filtering: Data not found in results dict.")
            continue # Skip if fundamental data fetch failed or timed out

        print(f"[{ticker}] Applying fundamental filters...")
        try:
            fund_data = fund_results[ticker_symbol]
            ni, prev_ni, pe, pb = fund_data

            # Calculate growth (handle division by zero or invalid prev_ni)
            growth = np.nan # Default to NaN
            if prev_ni is not None and isinstance(prev_ni, (int, float)) and not np.isnan(prev_ni):
                 if ni is not None and isinstance(ni, (int, float)) and not np.isnan(ni):
                     if prev_ni != 0:
                         growth = ((ni - prev_ni) / abs(prev_ni) * 100)
                     elif ni > 0: # Growth from zero
                         growth = np.inf
                     elif ni < 0:
                         growth = -np.inf
                     else: # ni = 0, prev_ni = 0
                         growth = 0.0
                 else:
                     print(f"[{ticker}] Growth Calc Warning: Current NI is invalid ({ni}).")
            else:
                 print(f"[{ticker}] Growth Calc Warning: Previous NI is invalid ({prev_ni}).")

            print(f"[{ticker}] Calculated Growth: {growth}")

            # Apply fundamental filters, logging reasons for exclusion
            reason = None
            if ni < min_ni:
                reason = f"NI {ni:.2f}T < {min_ni:.1f}T"
            # Only filter by PE if pe is valid and max_pe is restrictive
            elif pe is not None and max_pe < 50.0 and pe > max_pe:
                 reason = f"P/E {pe:.1f} > {max_pe:.1f}"
            # Only filter by PB if pb is valid and max_pb is restrictive
            elif pb is not None and max_pb < 5.0 and pb > max_pb:
                 reason = f"P/B {pb:.1f} > {max_pb:.1f}"
            # Only filter by growth if growth is valid and min_growth is restrictive
            elif not pd.isna(growth) and min_growth > -100.0 and growth < min_growth:
                 reason = f"Growth {growth:.1f}% < {min_growth:.1f}%"
            elif pd.isna(growth) and min_growth > -100.0:
                 reason = f"Growth calculation failed (NaN)"

            if reason:
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"Filtered out: {reason}"
                print(f"[{ticker}] Filtering out (Fundamental): {reason}")
                continue # Skip this stock

            # Add to final results
            if growth == np.inf: growth_display = "+Inf"
            elif growth == -np.inf: growth_display = "-Inf"
            elif pd.isna(growth): growth_display = "N/A"
            else: growth_display = f"{growth:.1f}"

            final_results.append([
                ticker_symbol,
                f"{ni:.2f}", # Format NI
                growth_display, # Use display string for growth
                f"{pe:.1f}" if pe is not None else "N/A", # Format PE or N/A
                f"{pb:.1f}" if pb is not None else "N/A", # Format PB or N/A
                f"{rsi:.1f}", # Format RSI
                signal,
                rsi_history, # Keep history for charts
                growth # Keep original growth for potential sorting later if needed
            ])
            print(f"[{ticker}] Passed fundamental filters.")

        except Exception as e:
            error_msg = f"Fund. Apply Error: {e}\n{traceback.format_exc()}"
            print(f"[{ticker}] !!! EXCEPTION applying fundamental filters: {error_msg}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = error_msg

    print(f"Fundamental filtering complete. {len(final_results)} stocks passed.")
    return final_results


@st.cache_data(ttl=300)
def create_rsi_chart_image(rsi_values, current_rsi, ticker="N/A"):
    '''Create a matplotlib chart for RSI values and return as image bytes'''
    print(f"[{ticker}] Creating RSI chart image...")
    if isinstance(rsi_values, list):
        rsi_values = np.array(rsi_values)

    if rsi_values is None or len(rsi_values) == 0:
        print(f"[{ticker}] Chart Error: No RSI data provided.")
        fig, ax = plt.subplots(figsize=(3, 1.5))
        ax.text(0.5, 0.5, "No RSI Data", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

    try:
        fig, ax = plt.subplots(figsize=(3, 1.5))
        x = range(len(rsi_values))
        ax.plot(x, rsi_values, color='blue', linewidth=1.5)
        ax.axhline(y=OVERBOUGHT_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=OVERSOLD_THRESHOLD, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.fill_between(x, OVERBOUGHT_THRESHOLD, 100, color='red', alpha=0.1)
        ax.fill_between(x, 0, OVERSOLD_THRESHOLD, color='green', alpha=0.1)
        ax.set_ylim(0, 100)

        num_ticks = min(5, len(rsi_values))
        tick_indices = np.linspace(0, len(rsi_values) - 1, num_ticks, dtype=int)
        tick_labels = [f"D-{len(rsi_values)-1-i}" for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha='right')

        ax.set_yticks([0, OVERSOLD_THRESHOLD, 50, OVERBOUGHT_THRESHOLD, 100])
        ax.set_yticklabels(['0', str(OVERSOLD_THRESHOLD), '50', str(OVERBOUGHT_THRESHOLD), '100'], fontsize=8)

        text_x_pos = len(rsi_values) - 1
        text_y_pos = current_rsi + 5 if current_rsi < 95 else current_rsi - 5
        ax.text(text_x_pos, text_y_pos, f'{current_rsi:.1f}', verticalalignment='center', horizontalalignment='right', fontsize=9, color='black', fontweight='bold')
        ax.scatter(len(rsi_values)-1, current_rsi, color='blue', s=30, zorder=5)

        ax.set_title(f"RSI({RSI_PERIOD}) Chart", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        print(f"[{ticker}] RSI chart image created successfully.")
        return buf
    except Exception as e:
        error_msg = f"Chart Creation Error: {e}"
        print(f"[{ticker}] !!! EXCEPTION creating chart: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[f"{ticker}_chart"] = error_msg
        # Return a placeholder image or re-raise?
        fig, ax = plt.subplots(figsize=(3, 1.5))
        ax.text(0.5, 0.5, "Chart Error", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

def process_batch_technical_first(batch_tickers, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral, exchange):
    '''Process a batch of tickers with technical filters first (yfinance version).'''
    results = []
    print(f"--- Processing technical batch of {len(batch_tickers)} tickers ({batch_tickers[0]}...{batch_tickers[-1]}) ---")
    process_func = partial(
        process_ticker_technical_first,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        show_oversold=show_oversold,
        show_overbought=show_overbought,
        show_neutral=show_neutral,
        exchange=exchange
    )
    # Using ThreadPoolExecutor for I/O bound yfinance calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_func, ticker): ticker for ticker in batch_tickers}
        print(f"Submitted {len(future_to_ticker)} technical jobs to executor.")
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            processed_count += 1
            print(f"({processed_count}/{len(future_to_ticker)}) Future completed for {ticker}")
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(f"Result added for {ticker}")
            except Exception as exc:
                 error_msg = f'Tech. Batch Error processing future for {ticker}: {exc}'
                 print(f"!!! EXCEPTION {error_msg}")
                 st.session_state.setdefault("errors", {})
                 st.session_state.errors[ticker] = error_msg

    print(f"--- Technical batch processing complete. {len(results)} passed. ---")
    return results


def main():
    st.set_page_config(
        page_title="Multi-Exchange Stock Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS (minor adjustments if needed)
    st.markdown('''
    <style>
    /* Make header sticky */
    .stDataFrame th {
        position: sticky;
        top: 0;
        background: white; /* Match background */
        z-index: 1;
    }
    /* Add border to expanders */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stSidebar {
            width: 100%;
        }
        .stDataFrame {
            width: 100%;
            overflow-x: auto;
        }
    }
    .stProgress > div > div {
        height: 10px;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    ''', unsafe_allow_html=True)

    # Initialize session state
    if "errors" not in st.session_state: st.session_state.errors = {}
    if "warnings" not in st.session_state: st.session_state.warnings = {} # Added warnings log
    if "filtered_out_technical" not in st.session_state: st.session_state.filtered_out_technical = {}
    if "filtered_out_fundamental" not in st.session_state: st.session_state.filtered_out_fundamental = {}
    if "last_refresh" not in st.session_state: st.session_state.last_refresh = None
    if "results_cache" not in st.session_state: st.session_state.results_cache = None
    if "selected_exchange" not in st.session_state: st.session_state.selected_exchange = "IDX"
    if "filter_settings" not in st.session_state:
        st.session_state.filter_settings = {
            "rsi_min": 0, "rsi_max": 100,
            "show_oversold": True, "show_overbought": True, "show_neutral": True,
            "min_ni": DEFAULT_MIN_NI, # Use updated default
            "max_pe": DEFAULT_MAX_PE, # Use updated default
            "max_pb": DEFAULT_MAX_PB, # Use updated default
            "min_growth": DEFAULT_MIN_GROWTH # Use updated default
        }

    # Get exchange information
    exchange_info = get_exchange_info()
    
    # App header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Multi-Exchange Stock Screener")
        
        # Exchange selection
        exchange_options = list(exchange_info.keys())
        selected_exchange = st.selectbox(
            "Select Stock Exchange",
            exchange_options,
            index=exchange_options.index(st.session_state.selected_exchange),
            format_func=lambda x: f"{exchange_info[x]['name']} ({x})"
        )
        
        # Update session state if exchange changed
        if selected_exchange != st.session_state.selected_exchange:
            st.session_state.selected_exchange = selected_exchange
            st.session_state.results_cache = None  # Clear results cache when exchange changes
            
        # Get tickers for selected exchange
        exchange_tickers = get_exchange_tickers(selected_exchange)
        
        st.markdown(f"Screening **{len(exchange_tickers)}** stocks from **{exchange_info[selected_exchange]['name']}** (Technical first, then Fundamental)")
    
    with col2:
        st.metric(f"Total {selected_exchange} Stocks", f"{len(exchange_tickers)}")

    # Sidebar filters
    with st.sidebar:
        st.header("Screening Filters")
        tab1, tab2, tab3, tab4 = st.tabs(["Technical", "Fundamental", "Performance", "Settings"])

        with tab1:
            st.subheader("Technical Filters (First Pass)")
            st.caption(f"RSI Period: {RSI_PERIOD} days (Wilder's Smoothing)")
            rsi_range = st.slider("RSI Range", 0, 100, (st.session_state.filter_settings["rsi_min"], st.session_state.filter_settings["rsi_max"]), help="Filter stocks by RSI value range.")
            rsi_min, rsi_max = rsi_range
            show_oversold = st.checkbox("Show Oversold (RSI < 30)", st.session_state.filter_settings["show_oversold"])
            show_overbought = st.checkbox("Show Overbought (RSI > 70)", st.session_state.filter_settings["show_overbought"])
            show_neutral = st.checkbox("Show Neutral (30 <= RSI <= 70)", st.session_state.filter_settings["show_neutral"])

        with tab2:
            st.subheader("Fundamental Filters (Second Pass)")
            min_ni = st.number_input("Min Net Income (Trillion)", value=st.session_state.filter_settings["min_ni"], min_value=0.0, max_value=100.0, step=0.1, help="Minimum Net Income in trillion currency units.")
            max_pe = st.number_input("Max P/E Ratio", value=st.session_state.filter_settings["max_pe"], min_value=0.0, max_value=200.0, step=1.0, help="Maximum Price to Earnings ratio.")
            max_pb = st.number_input("Max P/B Ratio", value=st.session_state.filter_settings["max_pb"], min_value=0.0, max_value=20.0, step=0.1, help="Maximum Price to Book ratio.")
            min_growth = st.number_input("Min YoY Growth (%)", value=st.session_state.filter_settings["min_growth"], min_value=-100.0, max_value=100.0, step=5.0, help="Minimum Year-over-Year growth percentage.")

        with tab3:
            st.subheader("Performance Settings")
            max_workers = st.slider("Max Parallel Workers", 1, 20, MAX_WORKERS, help="Maximum number of parallel workers for data fetching.")
            batch_size = st.slider("Batch Size", 10, 100, BATCH_SIZE, help="Number of tickers to process in each batch.")
            max_tickers = st.slider("Max Tickers to Process", 100, len(exchange_tickers), min(MAX_TICKERS, len(exchange_tickers)), help="Maximum number of tickers to process.")

        with tab4:
            st.subheader("Debug Settings")
            show_debug_logs = st.checkbox("Show Debug Logs", False)
            clear_cache = st.button("Clear Cache")
            if clear_cache:
                st.cache_data.clear()
                st.session_state.results_cache = None
                st.success("Cache cleared!")

        # Run screener button
        run_screener = st.button("Run Screener Now")

    # Save filter settings to session state
    st.session_state.filter_settings = {
        "rsi_min": rsi_min, "rsi_max": rsi_max,
        "show_oversold": show_oversold, "show_overbought": show_overbought, "show_neutral": show_neutral,
        "min_ni": min_ni, "max_pe": max_pe, "max_pb": max_pb, "min_growth": min_growth
    }

    # Main content area
    st.header(f"{exchange_info[selected_exchange]['name']} Stock Screener")
    st.markdown(f"Screening {len(exchange_tickers)} stocks (Technical first, then Fundamental)")

    # Tabs for results and logs
    tab1, tab2 = st.tabs(["Screener Results", "About & Logs"])

    with tab1:
        # Check if we need to run the screener
        if run_screener or st.session_state.results_cache is None:
            # Reset logs
            st.session_state.errors = {}
            st.session_state.warnings = {}
            st.session_state.filtered_out_technical = {}
            st.session_state.filtered_out_fundamental = {}

            # Start time
            start_time = time.time()
            st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Limit number of tickers to process
            tickers_to_process = exchange_tickers[:max_tickers]
            
            # For IDX exchange, add .JK suffix to tickers
            if selected_exchange == 'IDX':
                tickers_to_process = [f"{ticker}.JK" for ticker in tickers_to_process]
            
            # Process in batches
            all_technical_results = []
            total_batches = (len(tickers_to_process) + batch_size - 1) // batch_size
            
            for i in range(0, len(tickers_to_process), batch_size):
                batch = tickers_to_process[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
                technical_results = process_batch_technical_first(
                    batch, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral, selected_exchange
                )
                all_technical_results.extend(technical_results)
                
                # Update progress
                progress = min(0.5 * (i + batch_size) / len(tickers_to_process), 0.5)  # First half of progress bar
                progress_bar.progress(progress)

            # Technical screening complete
            status_text.text(f"Technical screening complete. {len(all_technical_results)} stocks passed. Applying fundamental filters...")
            
            # Apply fundamental filters
            final_results = apply_fundamental_filters(
                all_technical_results, min_ni, max_pe, max_pb, min_growth, selected_exchange
            )
            
            # Update progress to 100%
            progress_bar.progress(1.0)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            status_text.text(f"Screening complete in {elapsed_time:.2f} seconds. {len(final_results)} stocks passed all filters.")
            
            # Cache results
            st.session_state.results_cache = final_results
        else:
            # Use cached results
            final_results = st.session_state.results_cache
            status_text = st.empty()
            status_text.text(f"Using cached results from {st.session_state.last_refresh}. {len(final_results)} stocks passed all filters.")

        # Display screening progress
        total_tickers = len(exchange_tickers)
        technical_passed = len(st.session_state.results_cache) if st.session_state.results_cache else 0
        
        # Progress bar for screening completion
        st.markdown(f"Screening Complete ({technical_passed} passed all filters)")
        screening_progress = st.progress(1.0 if technical_passed > 0 else 0.0)
        
        # Display results if any
        if final_results:
            # Create DataFrame for display
            df = pd.DataFrame(final_results, columns=[
                "Ticker", "Net Income (T)", "Growth (%)", "P/E", "P/B", "RSI", "Signal", "RSI History", "Growth Raw"
            ])
            
            # Drop the raw growth column used for sorting
            df = df.drop(columns=["Growth Raw"])
            
            # Display results in a table
            st.subheader(f"Technical & Fundamental Screening Results ({len(df)} Stocks)")
            
            # Create a grid layout for the results
            cols = st.columns(3)
            
            for i, row in enumerate(df.itertuples()):
                col_idx = i % 3
                with cols[col_idx]:
                    with st.expander(f"{row.Ticker} - RSI: {row.RSI} ({row.Signal})"):
                        # Display RSI chart
                        rsi_history = row._7  # RSI History is at index 7
                        rsi_value = float(row.RSI)
                        chart_img = create_rsi_chart_image(rsi_history, rsi_value, row.Ticker)
                        st.image(chart_img, caption=f"RSI({RSI_PERIOD}) Chart", use_column_width=True)
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Net Income", f"{row._2} T")
                            st.metric("P/E Ratio", row._4)
                        with col2:
                            st.metric("Growth", f"{row._3}%")
                            st.metric("P/B Ratio", row._5)
        else:
            st.warning("No stocks found matching all criteria. Try relaxing the filters (especially Fundamental filters like P/E, P/B, Net Income) or check the Debug Logs in the 'About & Logs' tab for potential data issues.")

    with tab2:
        st.subheader("About & Logs")
        
        # About section
        with st.expander("About This Screener"):
            st.markdown("""
            ### Multi-Exchange Stock Screener
            
            This application screens stocks from multiple exchanges using a two-pass approach:
            1. **Technical Screening**: Filters stocks based on RSI (Relative Strength Index) values and signals.
            2. **Fundamental Screening**: Further filters stocks that passed technical screening based on financial metrics.
            
            #### Supported Exchanges
            - **IDX**: Indonesia Stock Exchange
            - **NYSE**: New York Stock Exchange
            - **NASDAQ**: NASDAQ Stock Exchange
            - **AMEX**: American Stock Exchange
            
            #### Data Sources
            - All data is fetched from Yahoo Finance using the `yfinance` library.
            - Technical data is cached for 5 minutes.
            - Fundamental data is cached for 24 hours.
            
            #### Technical Indicators
            - **RSI (Relative Strength Index)**: Calculated using Wilder's smoothing method with a period of 25 days.
            - **Signals**: Oversold (RSI < 30), Overbought (RSI > 70), Neutral (30 <= RSI <= 70).
            
            #### Fundamental Metrics
            - **Net Income**: Current net income in trillion currency units.
            - **Growth**: Year-over-Year growth percentage of net income.
            - **P/E Ratio**: Price to Earnings ratio.
            - **P/B Ratio**: Price to Book ratio.
            """)
        
        # Show logs if debug is enabled
        if show_debug_logs:
            # Errors
            with st.expander(f"Errors (Failed Operations) - {len(st.session_state.errors)} items"):
                if st.session_state.errors:
                    for ticker, error in st.session_state.errors.items():
                        st.error(f"{ticker}: {error}")
                else:
                    st.success("No errors logged.")
            
            # Warnings
            with st.expander(f"Warnings (Data Issues) - {len(st.session_state.warnings)} items"):
                if st.session_state.warnings:
                    for ticker, warning in st.session_state.warnings.items():
                        st.warning(f"{ticker}: {warning}")
                else:
                    st.success("No warnings logged.")
            
            # Filtered out technical
            with st.expander(f"Filtered Out (Technical) - {len(st.session_state.filtered_out_technical)} items"):
                if st.session_state.filtered_out_technical:
                    for ticker, reason in st.session_state.filtered_out_technical.items():
                        st.info(f"{ticker}: {reason}")
                else:
                    st.success("No stocks logged as filtered out by technical criteria.")
            
            # Filtered out fundamental
            with st.expander(f"Filtered Out (Fundamental) - {len(st.session_state.filtered_out_fundamental)} items"):
                if st.session_state.filtered_out_fundamental:
                    for ticker, reason in st.session_state.filtered_out_fundamental.items():
                        st.info(f"{ticker}: {reason}")
                else:
                    st.success("No stocks logged as filtered out by fundamental criteria.")


if __name__ == "__main__":
    main()
