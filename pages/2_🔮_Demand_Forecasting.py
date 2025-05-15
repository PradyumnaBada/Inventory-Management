# pages/2_🔮_Demand_Forecasting.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from forecasting_engine import train_forecast_prophet_with_regressors
from data_generation_engine import generate_future_features_for_prophet # Make sure this function is correctly imported if it's in data_generation_engine.py
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.markdown("# 🔮 Demand Forecasting with Prophet & User-Defined Future Features")
st.sidebar.header("Demand Forecasting")

if 'historical_df' not in st.session_state or \
   st.session_state.historical_df is None or \
   st.session_state.historical_df.empty:
    st.warning("Please generate or load historical data on the '📈 Historical Data and EDA' page first.")
    st.stop()

historical_df_for_forecast = st.session_state.historical_df.copy()
last_historical_date = pd.to_datetime(historical_df_for_forecast['date'].max())

st.subheader("Forecasting Parameters")
forecast_horizon_months = st.slider("Select Forecast Horizon (Months):", min_value=1, max_value=12, value=3, step=1)
forecast_horizon_days = forecast_horizon_months * 30 

# --- User Interface for Defining Future Features ---
st.sidebar.subheader("Define Future Feature Scenarios for Forecast")
if 'future_promo_plans' not in st.session_state: st.session_state.future_promo_plans = []
if 'future_event_plans' not in st.session_state: st.session_state.future_event_plans = []

with st.sidebar.expander("Plan Future Promotions", expanded=True): # Expanded default for easier use
    f_promo_start = st.date_input("Promotion Start Date", value=last_historical_date + timedelta(days=max(1, forecast_horizon_days // 3)), min_value=last_historical_date + timedelta(days=1), key="f_promo_start")
    f_promo_end = st.date_input("Promotion End Date", value=f_promo_start + timedelta(days=13), min_value=f_promo_start, key="f_promo_end")
    f_promo_discount = st.slider("Price Discount % for this Promo", 0.0, 0.5, 0.20, 0.01, format="%.2f", key="f_promo_discount")
    if st.button("Add Future Promotion", key="add_f_promo"):
        if f_promo_end < f_promo_start: st.error("End date before start.")
        else:
            st.session_state.future_promo_plans.append({'start': pd.to_datetime(f_promo_start), 'end': pd.to_datetime(f_promo_end), 'discount': f_promo_discount})
            st.success("Future promotion added.")
    if st.session_state.future_promo_plans:
        st.write("Current Future Promotion Plans:"); [st.caption(f"- {p['discount']*100:.0f}%: {p['start']:%Y-%m-%d} to {p['end']:%Y-%m-%d}") for p in st.session_state.future_promo_plans]
        if st.button("Clear Future Promotions", key="clear_f_promo"): st.session_state.future_promo_plans = []

with st.sidebar.expander("Plan Future Seasonal Events", expanded=True): # Expanded default
    f_event_start = st.date_input("Event Start Date", value=last_historical_date + timedelta(days=max(1, forecast_horizon_days // 2)), min_value=last_historical_date + timedelta(days=1), key="f_event_start")
    f_event_end = st.date_input("Event End Date", value=f_event_start + timedelta(days=20), min_value=f_event_start, key="f_event_end")
    f_event_multiplier = st.slider("Seasonal Event Multiplier", 0.5, 2.5, 1.5, 0.1, format="%.1f", key="f_event_multi")
    if st.button("Add Future Event", key="add_f_event"):
        if f_event_end < f_event_start: st.error("End date before start.")
        else:
            st.session_state.future_event_plans.append({'start': pd.to_datetime(f_event_start), 'end': pd.to_datetime(f_event_end), 'multiplier': f_event_multiplier})
            st.success("Future seasonal event added.")
    if st.session_state.future_event_plans:
        st.write("Current Future Event Plans:"); [st.caption(f"- x{p['multiplier']:.1f}: {p['start']:%Y-%m-%d} to {p['end']:%Y-%m-%d}") for p in st.session_state.future_event_plans]
        if st.button("Clear Future Events", key="clear_f_event"): st.session_state.future_event_plans = []

regressor_cols_list = ['price_discount_percentage', 'seasonal_event_multiplier']

if st.button("Generate Demand Forecast (Prophet with User-Defined Future Features)"):
    with st.spinner(f"Generating forecast for {forecast_horizon_months} months..."):
        future_features_df = generate_future_features_for_prophet( # Ensure this func is correctly imported
            last_historical_date, 
            forecast_horizon_days,
            user_promo_plans=st.session_state.future_promo_plans,
            user_event_plans=st.session_state.future_event_plans
        )
        st.session_state.final_future_features_df_prophet = future_features_df # For display

        try:
            # Ensure train_forecast_prophet_with_regressors returns the raw Prophet forecast too
            renamed_forecast_df, metrics, prophet_model_fitted, raw_prophet_forecast_obj = train_forecast_prophet_with_regressors(
                historical_df=historical_df_for_forecast, 
                future_features_df=future_features_df.copy(),
                forecast_horizon_days=forecast_horizon_days,
                target_col='demand',
                regressor_cols=regressor_cols_list,
                holidays_df=None 
            )
            st.session_state.prophet_forecast_df_renamed = renamed_forecast_df # Stores df with 'date', 'forecasted_demand'
            st.session_state.raw_prophet_forecast_for_plotting = raw_prophet_forecast_obj # Stores df with 'ds', 'yhat'
            st.session_state.prophet_metrics = metrics
            st.session_state.prophet_model_fitted = prophet_model_fitted
            st.success(f"Prophet forecast generated!")

        except Exception as e:
            st.error(f"Error during Prophet forecasting: {e}")
            # Clear potentially inconsistent states on error
            for key in ['prophet_forecast_df_renamed', 'raw_prophet_forecast_for_plotting', 'prophet_metrics', 'prophet_model_fitted']:
                if key in st.session_state: del st.session_state[key]


# This outer conditional block ensures that all forecast-related objects are available
if 'prophet_model_fitted' in st.session_state and \
   'raw_prophet_forecast_for_plotting' in st.session_state and \
   'prophet_forecast_df_renamed' in st.session_state:
    
    prophet_model_to_plot = st.session_state.prophet_model_fitted
    df_for_prophet_native_plot = st.session_state.raw_prophet_forecast_for_plotting # Use this for Prophet's plots
    forecast_display_df_renamed = st.session_state.prophet_forecast_df_renamed # Use this for simulation series

    metrics = st.session_state.prophet_metrics

    st.markdown("---")
    st.subheader("Prophet Forecast Results")

    hist_to_plot = historical_df_for_forecast.copy()
    if 'date' not in hist_to_plot.columns and isinstance(hist_to_plot.index, pd.DatetimeIndex):
        hist_to_plot = hist_to_plot.reset_index()
    
    fig_fc = plot_plotly(prophet_model_to_plot, df_for_prophet_native_plot) 
    fig_fc.add_trace(go.Scatter(x=pd.to_datetime(hist_to_plot['date']), y=hist_to_plot['demand'], 
                                mode='lines', name='Historical Actuals', line=dict(color='blue')))
    fig_fc.update_layout(title='Historical Demand & Prophet Forecast (with User-Defined Future Features)',
                         xaxis_title='Date', yaxis_title='Demand')
    st.plotly_chart(fig_fc, use_container_width=True)

    st.write("#### Forecast Metrics (In-sample Fit):")
    if metrics: 
        st.write(f"- RMSE: {metrics.get('rmse', 'N/A'):.2f}", f" MAPE: {metrics.get('mape', 'N/A'):.2f}%" if pd.notna(metrics.get('mape')) else "")
    else: st.write("Metrics not available.")

    st.write("#### Prophet Model Components (Including Regressors):")
    try:
        fig_comp = plot_components_plotly(prophet_model_to_plot, df_for_prophet_native_plot)
        st.plotly_chart(fig_comp, use_container_width=True)
    except Exception as e: st.warning(f"Could not plot Prophet components: {e}")

    st.write("#### User-Defined Future Features Used for This Forecast:")
    if 'final_future_features_df_prophet' in st.session_state and st.session_state.final_future_features_df_prophet is not None:
        st.dataframe(st.session_state.final_future_features_df_prophet.head())
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            fig_f1 = px.line(st.session_state.final_future_features_df_prophet, x='date', y='price_discount_percentage', title='Future Price Discount %')
            st.plotly_chart(fig_f1, use_container_width=True)
        with col_feat2:
            fig_f2 = px.line(st.session_state.final_future_features_df_prophet, x='date', y='seasonal_event_multiplier', title='Future Seasonal Event Multiplier')
            st.plotly_chart(fig_f2, use_container_width=True)

    # Set the demand series for simulation using the DataFrame that has 'forecasted_demand'
    st.session_state.demand_series_for_simulation = forecast_display_df_renamed['forecasted_demand'].values.tolist()
    st.success("Forecasted demand is ready. Proceed to 'Inventory Simulation' page.")
else:
    if st.session_state.get('historical_df') is not None : # Only show this if hist data exists
        st.info("Define future feature plans and click 'Generate Demand Forecast'.")