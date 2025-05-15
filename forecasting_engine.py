# forecasting_engine.py
import pandas as pd
from prophet import Prophet
import numpy as np

def train_forecast_prophet_with_regressors(historical_df, 
                                           future_features_df, 
                                           forecast_horizon_days, # Can be derived from future_features_df
                                           target_col='demand',
                                           regressor_cols=None,
                                           holidays_df=None):
    if historical_df is None or historical_df.empty:
        raise ValueError("Historical data cannot be empty.")
    if 'date' not in historical_df.columns or target_col not in historical_df.columns:
        raise ValueError(f"Historical data must contain 'date' and '{target_col}' columns.")
    if regressor_cols:
        for col in regressor_cols:
            if col not in historical_df.columns:
                raise ValueError(f"Historical data missing regressor column: {col}")
            if future_features_df is None or col not in future_features_df.columns: # Check future_features_df too
                raise ValueError(f"Future features data missing regressor column: {col}")

    prophet_hist_df = historical_df.rename(columns={'date': 'ds', target_col: 'y'})
    prophet_hist_df['ds'] = pd.to_datetime(prophet_hist_df['ds'])

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

    if regressor_cols:
        for regressor in regressor_cols:
            model.add_regressor(regressor)
    
    if holidays_df is not None and not holidays_df.empty:
        if not all(col in holidays_df.columns for col in ['holiday', 'ds']):
            raise ValueError("Holidays DataFrame must contain 'holiday' and 'ds' columns.")
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        model.add_holidays(holidays_df)
    
    model.fit(prophet_hist_df)

    # Prepare future dataframe for prediction
    # It must include 'ds' and all regressor columns
    if future_features_df is None or future_features_df.empty :
        raise ValueError("Future features DataFrame cannot be empty when using regressors.")

    future_df_for_predict = future_features_df.rename(columns={'date': 'ds'})
    future_df_for_predict['ds'] = pd.to_datetime(future_df_for_predict['ds'])
    
    # Ensure future_df_for_predict covers the intended forecast horizon
    # and aligns with make_future_dataframe if we still wanted to use it.
    # For this setup, future_features_df *defines* the prediction period.
    if len(future_df_for_predict) != forecast_horizon_days:
        st.warning(f"Length of future features ({len(future_df_for_predict)}) does not match forecast horizon ({forecast_horizon_days}). Adjusting prediction frame.")
        # Make a frame of the correct length first
        future_dates_template = model.make_future_dataframe(periods=forecast_horizon_days, freq='D')
        # Merge, keeping only dates in template, then fill NaNs if any
        merged_future_df = pd.merge(future_dates_template[['ds']], future_df_for_predict, on='ds', how='left')
        if regressor_cols:
            for col in regressor_cols:
                merged_future_df[col] = merged_future_df[col].fillna(method='ffill').fillna(method='bfill')
                if merged_future_df[col].isnull().any():
                    st.error(f"Feature '{col}' still has NaNs for future. Using 0 as fallback.")
                    merged_future_df[col] = merged_future_df[col].fillna(0) # Fallback
        future_df_for_predict = merged_future_df

    
    forecast = model.predict(future_df_for_predict)

    forecast_to_return = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_to_return.rename(columns={'ds': 'date', 'yhat': 'forecasted_demand', 
                                       'yhat_lower': 'forecast_lower', 'yhat_upper': 'forecast_upper'}, inplace=True)
    
    for col_name in ['forecasted_demand', 'forecast_lower', 'forecast_upper']:
        forecast_to_return[col_name] = np.maximum(0, forecast_to_return[col_name]).round().astype(int)

    metrics = {}
    try:
        fitted_values_on_hist = forecast[forecast['ds'].isin(prophet_hist_df['ds'])]
        if not fitted_values_on_hist.empty:
            comparison_df = pd.merge(prophet_hist_df[['ds', 'y']], fitted_values_on_hist[['ds', 'yhat']], on='ds', how='inner')
            if not comparison_df.empty:
                y_true = comparison_df['y']; y_pred = comparison_df['yhat']
                metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred)**2))
                mask_nonzero = y_true != 0
                if np.any(mask_nonzero): mape_values = np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero]); metrics['mape'] = np.mean(mape_values) * 100
                else: metrics['mape'] = np.nan
            else: metrics['rmse'] = np.nan; metrics['mape'] = np.nan
        else: metrics['rmse'] = np.nan; metrics['mape'] = np.nan
    except Exception as e: print(f"Error calculating in-sample metrics: {e}"); metrics['rmse'] = np.nan; metrics['mape'] = np.nan
        
    return forecast_to_return, metrics, model, forecast 