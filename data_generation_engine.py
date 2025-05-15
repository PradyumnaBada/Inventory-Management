# data_generation_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
# No direct Streamlit import needed here for core logic,
# logging will be returned to the calling page.

PERSISTENT_DATA_DIR = "app_data"
HISTORICAL_DATA_FILE = os.path.join(PERSISTENT_DATA_DIR, "persistent_historical_data.csv")
METADATA_FILE = os.path.join(PERSISTENT_DATA_DIR, "app_metadata.json")
INITIAL_HISTORY_YEARS = 5

def _generate_feature_patterns(df_segment_in):
    df_segment = df_segment_in.copy()
    df_segment['price_discount_percentage'] = 0.0
    for year_val in df_segment['date'].dt.year.unique():
        # Ensure start_date is within the segment for correct year handling
        # This logic for placing promos needs to be robust if segments are small (e.g. few days)
        # For simplicity, we'll assume segments are reasonably large or this logic gets more complex
        # to ensure promos fall within the segment if desired.
        try:
            mid_year_sale_start = pd.Timestamp(year=year_val, month=7, day=np.random.randint(1,8))
            mid_year_sale_end = mid_year_sale_start + timedelta(days=13)
            df_segment.loc[(df_segment['date'] >= mid_year_sale_start) & (df_segment['date'] <= mid_year_sale_end), 'price_discount_percentage'] = 0.20

            holiday_sale_start = pd.Timestamp(year=year_val, month=11, day=np.random.randint(15,22))
            holiday_sale_end = holiday_sale_start + timedelta(days=20)
            df_segment.loc[(df_segment['date'] >= holiday_sale_start) & (df_segment['date'] <= holiday_sale_end), 'price_discount_percentage'] = 0.30
            
            for _ in range(np.random.randint(1,3)): # 1 or 2 small promos per year
                promo_doy = np.random.randint(1, 360) # day of year for promo start
                # Create date based on year_val to ensure it's in the correct year scope
                small_promo_start_this_year = pd.Timestamp(year=year_val, month=1, day=1) + timedelta(days=promo_doy-1)
                small_promo_end_this_year = small_promo_start_this_year + timedelta(days=np.random.randint(4,7)) # 5-7 days long
                
                if small_promo_start_this_year.year != year_val: # Basic check
                    continue

                # Ensure promo dates are within the segment being processed
                segment_min_date = df_segment['date'].min()
                segment_max_date = df_segment['date'].max()
                
                actual_promo_start = max(small_promo_start_this_year, segment_min_date)
                actual_promo_end = min(small_promo_end_this_year, segment_max_date)

                if actual_promo_start <= actual_promo_end:
                    mask = (df_segment['date'] >= actual_promo_start) & \
                           (df_segment['date'] <= actual_promo_end) & \
                           (df_segment['price_discount_percentage'] == 0.0) # Avoid overlap
                    df_segment.loc[mask, 'price_discount_percentage'] = np.random.choice([0.05, 0.10, 0.15])
        except ValueError as e: # Handles potential date issues like Feb 29
            print(f"Warning: Skipping a promo generation for year {year_val} due to date issue: {e}")
            pass

    df_segment['seasonal_event_multiplier'] = 1.0
    df_segment.loc[df_segment['date'].dt.month.isin([6, 7, 8]), 'seasonal_event_multiplier'] = 1.3
    df_segment.loc[(df_segment['date'].dt.month == 12) & (df_segment['date'].dt.day.isin(range(10,25))), 'seasonal_event_multiplier'] = 1.6
    df_segment.loc[(df_segment['date'].dt.month == 11) & (df_segment['date'].dt.day.isin(range(20,31))), 'seasonal_event_multiplier'] = 1.4
    df_segment.loc[df_segment['date'].dt.month.isin([1, 2]), 'seasonal_event_multiplier'] = 0.90
    return df_segment

def generate_data_for_period(start_date_dt, end_date_dt, fixed_trend_start_ref_date):
    if start_date_dt > end_date_dt:
        return pd.DataFrame(columns=['date', 'demand', 'price_discount_percentage', 'seasonal_event_multiplier'])
    date_rng = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
    n_days = len(date_rng)
    if n_days == 0: return pd.DataFrame(columns=['date', 'demand', 'price_discount_percentage', 'seasonal_event_multiplier'])
    
    df = pd.DataFrame({'date': date_rng})
    df = _generate_feature_patterns(df) # Generate features for this specific segment

    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    
    df['days_from_ref_start'] = (df['date'] - fixed_trend_start_ref_date).dt.days

    base_demand = 60
    trend_per_day = 20 / (INITIAL_HISTORY_YEARS * 365) # Overall trend slope
    trend_factor = df['days_from_ref_start'] * trend_per_day

    yearly_seasonality = 20 * np.sin(2 * np.pi * (df['day_of_year'] - 90) / 365.25)
    weekly_seasonality = (8 * (df['day_of_week'] == 4) + 12 * (df['day_of_week'] == 5) + \
                          6 * (df['day_of_week'] == 6) - 7 * (df['day_of_week'] == 1))
    price_effect_strength = 120
    noise = np.random.normal(0, 6, n_days)
    demand_before_event_multiplier = (base_demand + trend_factor + yearly_seasonality + weekly_seasonality + \
                                     (df['price_discount_percentage'] * price_effect_strength))
    df['demand'] = (demand_before_event_multiplier * df['seasonal_event_multiplier'] + noise)
    df['demand'] = np.maximum(5, df['demand']).round().astype(int)
    return df[['date', 'demand', 'price_discount_percentage', 'seasonal_event_multiplier']]

def ensure_data_dir():
    if not os.path.exists(PERSISTENT_DATA_DIR):
        try: os.makedirs(PERSISTENT_DATA_DIR)
        except OSError as e: print(f"Warning: Could not create data directory '{PERSISTENT_DATA_DIR}': {e}")


def load_or_generate_historical_data(): # Removed effective_current_date_dt parameter
    """
    Manages loading, appending, and generating historical data up to the ACTUAL CURRENT DATE.
    Returns the historical DataFrame and a list of log messages.
    """
    ensure_data_dir()
    
    # Use the actual current date for this run
    actual_current_date = pd.to_datetime(datetime.now().date()) # Use .date() to ignore time part for daily data

    last_data_date_from_file = None
    historical_df = None
    generation_log_messages = []

    # Define a fixed reference start date for consistent trend calculation across all runs
    # This should be set once, e.g., 5 years before a fixed "project epoch" or first possible run date
    # For this demo, let's base it on a fixed past date so trend doesn't restart with every append.
    PROJECT_EPOCH_FOR_TREND = pd.to_datetime("2020-01-01") # An arbitrary fixed start for trend calc
    FIXED_TREND_START_REF_DATE = PROJECT_EPOCH_FOR_TREND - timedelta(days=INITIAL_HISTORY_YEARS * 365)


    if os.path.exists(METADATA_FILE) and os.path.exists(HISTORICAL_DATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                last_data_date_from_file = pd.to_datetime(metadata.get('last_data_date'))
            
            historical_df = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=['date'])
            generation_log_messages.append(f"Loaded existing data up to {last_data_date_from_file.strftime('%Y-%m-%d') if last_data_date_from_file else 'N/A'}.")
        except Exception as e:
            generation_log_messages.append(f"Error loading existing data/metadata: {e}. Will regenerate.")
            last_data_date_from_file = None 
            historical_df = None

    if last_data_date_from_file is None: 
        generation_log_messages.append("No valid existing data. Performing initial 5-year data generation.")
        gen_end_date = actual_current_date
        gen_start_date = gen_end_date - timedelta(days=INITIAL_HISTORY_YEARS * 365 - 1) # -1 for inclusive end date
        
        historical_df = generate_data_for_period(gen_start_date, gen_end_date, FIXED_TREND_START_REF_DATE)
        
        if not historical_df.empty:
            historical_df.to_csv(HISTORICAL_DATA_FILE, index=False)
            with open(METADATA_FILE, 'w') as f:
                json.dump({'last_data_date': gen_end_date.strftime('%Y-%m-%d')}, f)
            generation_log_messages.append(f"Initial {INITIAL_HISTORY_YEARS}-year data generated and saved up to {gen_end_date.strftime('%Y-%m-%d')}.")
        else: # Fallback
            generation_log_messages.append("Error: Initial data generation resulted in empty. Using minimal dummy.")
            dummy_dates = pd.date_range(end=actual_current_date, periods=max(1, INITIAL_HISTORY_YEARS * 365), freq='D')
            historical_df = pd.DataFrame({'date': dummy_dates, 'demand': np.random.randint(10, 50, size=len(dummy_dates)), 'price_discount_percentage': 0.0, 'seasonal_event_multiplier': 1.0})

    elif last_data_date_from_file < actual_current_date:
        append_start_date = last_data_date_from_file + timedelta(days=1)
        append_end_date = actual_current_date # Generate up to today
        
        if append_start_date <= append_end_date: 
            generation_log_messages.append(f"Appending data from {append_start_date.strftime('%Y-%m-%d')} to {append_end_date.strftime('%Y-%m-%d')}.")
            new_data_df = generate_data_for_period(append_start_date, append_end_date, FIXED_TREND_START_REF_DATE)
            
            if not new_data_df.empty:
                historical_df = pd.concat([historical_df, new_data_df], ignore_index=True)
                historical_df.sort_values(by='date', inplace=True) 
                historical_df.drop_duplicates(subset=['date'], keep='last', inplace=True) 
                historical_df.to_csv(HISTORICAL_DATA_FILE, index=False)
                with open(METADATA_FILE, 'w') as f:
                    json.dump({'last_data_date': append_end_date.strftime('%Y-%m-%d')}, f)
                generation_log_messages.append(f"Appended data. New last date: {append_end_date.strftime('%Y-%m-%d')}.")
            else:
                generation_log_messages.append("No new data generated for the append period (period valid, generation empty).")
        else:
            generation_log_messages.append(f"Data is current ({last_data_date_from_file.strftime('%Y-%m-%d')}), no append needed.")
    
    elif historical_df is None : 
        generation_log_messages.append("Critical error: Historical DataFrame is None after processing. Using minimal dummy.")
        dummy_dates = pd.date_range(end=actual_current_date, periods=10, freq='D')
        historical_df = pd.DataFrame({'date': dummy_dates, 'demand': 10, 'price_discount_percentage': 0, 'seasonal_event_multiplier':1})
    else: 
        generation_log_messages.append(f"Data is current ({last_data_date_from_file.strftime('%Y-%m-%d') if last_data_date_from_file else 'N/A'}).")

    return historical_df, generation_log_messages


# In data_generation_engine.py

def generate_future_features_for_prophet(historical_df_end_date, forecast_horizon_days, user_promo_plans=None, user_event_plans=None):
    """
    Generates future feature values for Prophet based on user inputs or default patterns.
    """
    forecast_start_date = historical_df_end_date + timedelta(days=1)
    future_date_rng = pd.date_range(start=forecast_start_date, periods=forecast_horizon_days, freq='D')
    future_df = pd.DataFrame({'date': future_date_rng})

    # Initialize with defaults (no promo, standard event multiplier)
    future_df['price_discount_percentage'] = 0.0
    future_df['seasonal_event_multiplier'] = 1.0 # Start with a baseline

    # Apply general future seasonal event patterns (can be overridden by user plans)
    # This uses the _generate_feature_patterns to get a baseline seasonal structure for multipliers
    # but we will primarily rely on user input for specific values.
    # If no user input, these default patterns from _generate_feature_patterns will apply.
    temp_event_pattern_df = _generate_feature_patterns(future_df[['date']].copy()) 
    future_df['seasonal_event_multiplier'] = temp_event_pattern_df['seasonal_event_multiplier']
    # For price, we want it to be 0 unless explicitly set by the user for the future.
    future_df['price_discount_percentage'] = 0.0 # Reset after _generate_feature_patterns

    # Apply user-defined future promotion plans
    if user_promo_plans:
        for plan in user_promo_plans:
            mask = (future_df['date'] >= plan['start']) & (future_df['date'] <= plan['end'])
            future_df.loc[mask, 'price_discount_percentage'] = plan['discount']
    
    # Apply user-defined future seasonal event multiplier plans
    if user_event_plans:
        for plan in user_event_plans:
            mask = (future_df['date'] >= plan['start']) & (future_df['date'] <= plan['end'])
            future_df.loc[mask, 'seasonal_event_multiplier'] = plan['multiplier']
            
    return future_df[['date', 'price_discount_percentage', 'seasonal_event_multiplier']]


if __name__ == '__main__':
    print(f"Actual current date for tests: {datetime.now().strftime('%Y-%m-%d')}")
    
    # To simulate running on different days for testing append logic:
    test_date_1 = pd.to_datetime(datetime.now().date()) # Today
    test_date_2 = test_date_1 + timedelta(days=30)    # ~1 month later
    test_date_3 = test_date_1 - timedelta(days=10)   # A past date (for ensuring initial gen)

    # --- Test Scenario 1: First run (or run with a past effective date) ---
    print(f"\n--- TEST 1: Simulating First Run (or run with effective date {test_date_3.strftime('%Y-%m-%d')}) ---")
    if os.path.exists(HISTORICAL_DATA_FILE): os.remove(HISTORICAL_DATA_FILE)
    if os.path.exists(METADATA_FILE): os.remove(METADATA_FILE)
    
    # Forcing the "effective current date" to be in the past for the first test
    # In the actual app, load_or_generate_historical_data() uses datetime.now() internally
    # So, to test initial generation up to a point, we modify what it considers 'today'
    # For the standalone test, let's assume 'today' is test_date_1, and we call with it
    
    # Simulate what happens if the app is run on test_date_1
    df1, logs1 = load_or_generate_historical_data() # This will use datetime.now()
    # To test initial generation up to test_date_1, we'd need to mock datetime.now() or pass date.
    # The function is now designed to use datetime.now() implicitly.
    # So, for the test block, it will always try to generate up to the *actual* today.
    
    # Let's reset and test initial generation up to a specific past date for clarity
    if os.path.exists(HISTORICAL_DATA_FILE): os.remove(HISTORICAL_DATA_FILE)
    if os.path.exists(METADATA_FILE): os.remove(METADATA_FILE)
    
    # Save original datetime.now
    original_datetime_now = datetime.now
    class MockDateTime:
        @classmethod
        def now(cls): return original_datetime_now() - timedelta(days=30) # Simulate "today" was 30 days ago
    
    datetime_module = __import__('datetime')
    datetime_module.datetime = MockDateTime # Monkey patch (for testing only!)
    
    print(f"Testing initial generation as if 'today' was {(original_datetime_now() - timedelta(days=30)).strftime('%Y-%m-%d')}")
    df_initial, logs_initial = load_or_generate_historical_data()
    for log in logs_initial: print(f"Log: {log}")
    if not df_initial.empty: print(f"Initial DF shape: {df_initial.shape}, Last date: {df_initial['date'].max()}")

    # Restore datetime.now
    datetime_module.datetime = original_datetime_now

    # --- Test Scenario 2: Run again on actual today (should append) ---
    print(f"\n--- TEST 2: Simulating Run Again on actual 'today' (should append) ---")
    df_appended, logs_appended = load_or_generate_historical_data() # Uses actual datetime.now()
    for log in logs_appended: print(f"Log: {log}")
    if not df_appended.empty:
        print(f"Appended DF shape: {df_appended.shape}, Last date: {df_appended['date'].max()}")
        if not df_initial.empty : assert len(df_appended) >= len(df_initial) # Should have appended or be same if run on same day
        assert pd.to_datetime(df_appended['date'].max()).date() == datetime.now().date()
    
    # --- Test Scenario 3: Run again on actual today (should NOT append) ---
    print(f"\n--- TEST 3: Simulating Run Again on actual 'today' (should NOT append) ---")
    df_no_append, logs_no_append = load_or_generate_historical_data() # Uses actual datetime.now()
    for log in logs_no_append: print(f"Log: {log}")
    if not df_no_append.empty:
        print(f"No Append DF shape: {df_no_append.shape}, Last date: {df_no_append['date'].max()}")
        if not df_appended.empty : assert len(df_no_append) == len(df_appended)