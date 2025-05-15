# In pages/1_📈_Historical_Data_and_EDA.py

import streamlit as st
import pandas as pd
import plotly.express as px
from data_generation_engine import load_or_generate_historical_data # No more EFFECTIVE_CURRENT_DATE
from datetime import datetime # To display the actual current date

st.set_page_config(page_title="Historical Data", layout="wide")
st.markdown("# 📈 Historical Data Exploration & EDA")
st.sidebar.header("Historical Data")

# Display the actual current date the app is considering
current_processing_display_date = datetime.now().strftime('%Y-%m-%d')
st.info(f"Managing historical data up to: {current_processing_display_date}")

if 'historical_df' not in st.session_state or st.sidebar.button("Reload/Refresh Historical Data"):
    with st.spinner("Loading/Generating historical data... This may take a moment."):
        # Call the updated function, it now handles dates internally
        df, log_messages = load_or_generate_historical_data() 
        st.session_state.historical_df = df
        
        if 'generation_log' not in st.session_state:
            st.session_state.generation_log = []
        st.session_state.generation_log.extend(log_messages) # Append new logs
        
        # Display log messages from this run
        st.sidebar.subheader("Data Generation Log (Current Run):")
        if log_messages:
            for log_entry in log_messages:
                st.sidebar.caption(log_entry) 
        else:
            st.sidebar.caption("No new log messages from data generation.")
        st.session_state.data_loaded = True


if st.session_state.get('data_loaded', False) and \
   st.session_state.historical_df is not None and \
   not st.session_state.historical_df.empty:
    
    df_to_display = st.session_state.historical_df.copy()
    df_to_display['date'] = pd.to_datetime(df_to_display['date']) # Ensure it's datetime
    
    st.subheader(f"Data Preview (Ends: {df_to_display['date'].max().strftime('%Y-%m-%d')}, "
                 f"Total Days: {len(df_to_display)})")
    st.dataframe(df_to_display.tail(10))

    # ... rest of the EDA plots remain the same as in the previous version ...
    st.subheader("Descriptive Statistics")
    st.dataframe(df_to_display[['demand', 'price_discount_percentage', 'seasonal_event_multiplier']].describe())
    st.markdown("---")
    st.subheader("Visualizations")
    plot_df = df_to_display.copy()
    st.write("#### Demand Over Time")
    fig_demand = px.line(plot_df, x='date', y='demand', title='Historical Demand Over Time')
    st.plotly_chart(fig_demand, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Price Discount Percentage Over Time")
        fig_price = px.line(plot_df, x='date', y='price_discount_percentage', title='Price Discount % Over Time')
        fig_price.update_yaxes(range=[-0.05, max(0.05, plot_df['price_discount_percentage'].max() * 1.1)])
        st.plotly_chart(fig_price, use_container_width=True)
        st.write("#### Demand vs. Price Discount")
        fig_demand_price = px.scatter(plot_df, x='price_discount_percentage', y='demand', 
                                      title='Demand vs. Price Discount %', 
                                      labels={'price_discount_percentage':'Price Discount %'},
                                      trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig_demand_price, use_container_width=True)
    with col2:
        st.write("#### Seasonal Event Multiplier Over Time")
        fig_event = px.line(plot_df, x='date', y='seasonal_event_multiplier', title='Seasonal Event Multiplier Over Time')
        fig_event.update_yaxes(range=[min(0.95, plot_df['seasonal_event_multiplier'].min() * 0.9), max(1.05, plot_df['seasonal_event_multiplier'].max() * 1.1)])
        st.plotly_chart(fig_event, use_container_width=True)
        st.write("#### Demand vs. Seasonal Event Multiplier")
        fig_demand_event = px.scatter(plot_df, x='seasonal_event_multiplier', y='demand', 
                                       title='Demand vs. Seasonal Event Multiplier',
                                       labels={'seasonal_event_multiplier':'Seasonal Event Multiplier'},
                                       trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig_demand_event, use_container_width=True)
    st.success("Historical data processed. Proceed to 'Demand Forecasting'.")

elif not st.session_state.get('data_loaded', False):
    st.info("Click 'Reload/Refresh Historical Data' in the sidebar to start.")
else:
    st.error("Failed to load or generate historical data. Check console for errors from data_generation_engine.")