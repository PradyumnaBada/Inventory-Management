# app.py (Main application file using st.tabs, controls within tabs)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import engine functions
from data_generation_engine import load_or_generate_historical_data, generate_future_features_for_prophet
from forecasting_engine import train_forecast_prophet_with_regressors
from inventory_simulation_engine import run_simulation
from prophet.plot import plot_plotly, plot_components_plotly

# --- Page Configuration (do this once at the top) ---
st.set_page_config(
    page_title="Lindt Chocolate Inventory Suite",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse sidebar as we are not using it for nav/controls
)

# --- Initialize ALL Session State Variables Used Across Tabs ---
# This is crucial to prevent AttributeError on first run or after state clears
default_session_states = {
    'historical_df': None,
    'data_loaded': False,
    'generation_log': [],
    'future_promo_plans': [],
    'future_event_plans': [],
    'final_future_features_df_prophet': None,
    'prophet_forecast_df_renamed': None,
    'raw_prophet_forecast_for_plotting': None,
    'prophet_metrics': None,
    'prophet_model_fitted': None,
    'demand_series_for_simulation': [],
    'simulation_results': None,
    'final_demand_for_sim_display': [] # For displaying the actual demand fed to simulation
}
for key, default_value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --- Main App Title ---
st.title("🍫 Lindt Chocolate Inventory Management Suite")
st.caption(f"Application Data Managed up to: {datetime.now().strftime('%Y-%m-%d')}")

# --- Create Tabs for Navigation ---
tab_home, tab_eda, tab_forecast, tab_simulate = st.tabs([
    "🏠 Home",
    "📈 Historical Data & EDA",
    "🔮 Demand Forecasting",
    "⚙️ Inventory Simulation & Optimization"
])


# --- Tab 1: Home / Introduction ---
with tab_home:
    st.markdown("## Welcome to the Lindt Chocolate Inventory Suite!")
    st.markdown("""
    This tool helps you manage inventory for Lindt chocolate products by:
    1.  Exploring synthetically generated historical sales data (automatically updated to the current date).
    2.  Forecasting future demand based on this history and your planned promotions or anticipated seasonal events.
    3.  Simulating inventory strategies using these forecasts to optimize performance and understand trade-offs.

    **Data Handling:**
    - On the first run, 5 years of historical data ending today will be generated for a typical Lindt chocolate product.
    - On subsequent runs, the historical data will be automatically updated.
    - Features like 'price discounts' and 'seasonal events' influence the generated demand. You will define these for future periods on the Forecasting page.

    Select a tab above to begin.
    """)
    st.markdown("---")
    st.markdown("Developed for demonstration of integrated data generation, forecasting, and simulation.")


# --- Tab 2: Historical Data and EDA ---
with tab_eda:
    st.markdown("## 📈 Historical Data & EDA")
    
    st.markdown("### Data Management")
    if st.button("Load/Refresh Historical Data for Lindt Chocolate", key="load_hist_data_btn_main"):
        with st.spinner("Loading/Generating historical data..."):
            df, log_messages = load_or_generate_historical_data() 
            st.session_state.historical_df = df
            st.session_state.generation_log = log_messages 
            st.session_state.data_loaded = True
            # Clear downstream session state as historical data has changed
            for key_to_clear in ['prophet_forecast_df_renamed', 'raw_prophet_forecast_for_plotting', 
                        'prophet_metrics', 'prophet_model_fitted', 'demand_series_for_simulation',
                        'simulation_results', 'final_future_features_df_prophet', 
                        'future_promo_plans', 'future_event_plans', 'final_demand_for_sim_display']: 
                if key_to_clear in st.session_state: del st.session_state[key_to_clear]
            # Re-initialize lists that should be empty before next step
            st.session_state.future_promo_plans = [] 
            st.session_state.future_event_plans = []
            st.success("Historical data for Lindt chocolate refreshed.")

    if st.session_state.generation_log:
        with st.expander("View Data Generation Log", expanded=False):
            for log_entry in st.session_state.generation_log:
                st.caption(log_entry) 

    if st.session_state.get('data_loaded', False) and \
       st.session_state.historical_df is not None and \
       not st.session_state.historical_df.empty:
        
        df_to_display_eda = st.session_state.historical_df.copy()
        df_to_display_eda['date'] = pd.to_datetime(df_to_display_eda['date'])
        
        st.subheader(f"Data Preview (Ends: {df_to_display_eda['date'].max().strftime('%Y-%m-%d')}, Total Days: {len(df_to_display_eda)})")
        st.dataframe(df_to_display_eda.tail(10))
        st.subheader("Descriptive Statistics for Lindt Chocolate Demand & Features")
        st.dataframe(df_to_display_eda[['demand', 'price_discount_percentage', 'seasonal_event_multiplier']].describe())
        st.markdown("---")
        st.subheader("Visualizations")
        plot_df_eda = df_to_display_eda.copy()
        st.write("#### Lindt Chocolate Demand Over Time"); fig_demand_eda = px.line(plot_df_eda, x='date', y='demand', title='Historical Demand for Lindt Chocolate'); st.plotly_chart(fig_demand_eda, use_container_width=True)
        col1_eda, col2_eda = st.columns(2)
        with col1_eda:
            st.write("#### Price Discount % Over Time"); fig_price_eda = px.line(plot_df_eda, x='date', y='price_discount_percentage', title='Historical Price Discount %'); fig_price_eda.update_yaxes(range=[-0.05, max(0.05, plot_df_eda['price_discount_percentage'].max() * 1.1 if not plot_df_eda['price_discount_percentage'].empty else 0.5)]); st.plotly_chart(fig_price_eda, use_container_width=True)
            st.write("#### Demand vs. Price Discount"); fig_dp_eda = px.scatter(plot_df_eda, x='price_discount_percentage', y='demand', title='Demand vs. Price Discount %', trendline="ols", trendline_color_override="red"); st.plotly_chart(fig_dp_eda, use_container_width=True)
        with col2_eda:
            st.write("#### Seasonal Event Multiplier Over Time"); fig_event_eda = px.line(plot_df_eda, x='date', y='seasonal_event_multiplier', title='Historical Seasonal Event Multiplier'); fig_event_eda.update_yaxes(range=[min(0.95, plot_df_eda['seasonal_event_multiplier'].min() * 0.9 if not plot_df_eda['seasonal_event_multiplier'].empty else 0.9), max(1.05, plot_df_eda['seasonal_event_multiplier'].max() * 1.1 if not plot_df_eda['seasonal_event_multiplier'].empty else 1.1)]); st.plotly_chart(fig_event_eda, use_container_width=True)
            st.write("#### Demand vs. Seasonal Event Multiplier"); fig_de_eda = px.scatter(plot_df_eda, x='seasonal_event_multiplier', y='demand', title='Demand vs. Seasonal Event Multiplier', trendline="ols", trendline_color_override="red"); st.plotly_chart(fig_de_eda, use_container_width=True)
    elif not st.session_state.get('data_loaded', False):
        st.info("Click 'Load/Refresh Historical Data for Lindt Chocolate' above to begin.")
    else: st.error("Failed to load or generate historical data. Check logs or console.")


# --- Tab 3: Demand Forecasting ---
with tab_forecast:
    st.markdown("# 🔮 Demand Forecasting") # Updated

    if 'historical_df' not in st.session_state or st.session_state.historical_df is None or st.session_state.historical_df.empty:
        st.warning("Please generate/load historical data on the '📈 Historical Data & EDA' tab first.")
    else:
        historical_df_for_forecast = st.session_state.historical_df.copy()
        last_historical_date = pd.to_datetime(historical_df_for_forecast['date'].max())

        st.subheader("Forecasting Setup")
        fc_col_main1, fc_col_main2 = st.columns([1,3]) 

        with fc_col_main1:
            st.markdown("##### Forecast Horizon")
            forecast_horizon_months = st.slider("Months to Forecast:", min_value=1, max_value=12, value=3, step=1, key="fc_horizon_main")
            forecast_horizon_days = forecast_horizon_months * 30 
        
        with fc_col_main2:
            st.markdown("##### Define Future Scenarios for Lindt Chocolate Promotions & Events")
            with st.expander("Plan Future Promotions", expanded=False):
                p_col1, p_col2, p_col3, p_col4 = st.columns([2,2,1,1])
                with p_col1: f_promo_start = st.date_input("Start Date", value=last_historical_date + timedelta(days=max(1,30)), min_value=last_historical_date + timedelta(days=1), key="f_promo_start_main")
                with p_col2: f_promo_end = st.date_input("End Date", value=f_promo_start + timedelta(days=13), min_value=f_promo_start, key="f_promo_end_main")
                with p_col3: f_promo_discount = st.number_input("Discount %", 0.0, 0.5, 0.20, 0.01, format="%.2f", key="f_promo_discount_main")
                with p_col4: st.write(""); st.write(""); 
                if st.button("Add Promotion", key="add_f_promo_main"):
                    if f_promo_end < f_promo_start: st.error("End date before start.")
                    else: st.session_state.future_promo_plans.append({'start': pd.to_datetime(f_promo_start), 'end': pd.to_datetime(f_promo_end), 'discount': f_promo_discount})
                if st.session_state.future_promo_plans:
                    st.caption("Current Promotion Plans:"); [st.caption(f"- {p['discount']*100:.0f}%: {p['start']:%Y-%m-%d} to {p['end']:%Y-%m-%d}") for p in st.session_state.future_promo_plans]
                    if st.button("Clear Promotions", key="clear_f_promo_main"): st.session_state.future_promo_plans = []

            with st.expander("Plan Future Seasonal Events", expanded=False):
                e_col1, e_col2, e_col3, e_col4 = st.columns([2,2,1,1])
                with e_col1: f_event_start = st.date_input("Start Date", value=last_historical_date + timedelta(days=max(1,60)), min_value=last_historical_date + timedelta(days=1), key="f_event_start_main")
                with e_col2: f_event_end = st.date_input("End Date", value=f_event_start + timedelta(days=20), min_value=f_event_start, key="f_event_end_main")
                with e_col3: f_event_multiplier = st.number_input("Multiplier", 0.5, 2.5, 1.5, 0.1, format="%.1f", key="f_event_multi_main")
                with e_col4: st.write(""); st.write(""); 
                if st.button("Add Event", key="add_f_event_main"):
                    if f_event_end < f_event_start: st.error("End date before start.")
                    else: st.session_state.future_event_plans.append({'start': pd.to_datetime(f_event_start), 'end': pd.to_datetime(f_event_end), 'multiplier': f_event_multiplier})
            if st.session_state.future_event_plans:
                st.caption("Current Event Plans:"); [st.caption(f"- x{p['multiplier']:.1f}: {p['start']:%Y-%m-%d} to {p['end']:%Y-%m-%d}") for p in st.session_state.future_event_plans]
                if st.button("Clear Events", key="clear_f_event_main"): st.session_state.future_event_plans = []
        
        st.markdown("---") 
        if st.button("Generate Demand Forecast for Lindt Chocolate", key="gen_forecast_btn_main", type="primary"): # Updated
            with st.spinner(f"Generating forecast..."):
                future_features_df_fc = generate_future_features_for_prophet(
                    last_historical_date, forecast_horizon_days,
                    user_promo_plans=st.session_state.future_promo_plans,
                    user_event_plans=st.session_state.future_event_plans
                )
                st.session_state.final_future_features_df_prophet = future_features_df_fc
                try:
                    regressor_cols_list_fc = ['price_discount_percentage', 'seasonal_event_multiplier']
                    renamed_fc_df, mets, model_fit, raw_fc_obj = train_forecast_prophet_with_regressors(
                        historical_df=historical_df_for_forecast, future_features_df=future_features_df_fc.copy(),
                        forecast_horizon_days=forecast_horizon_days, target_col='demand',
                        regressor_cols=regressor_cols_list_fc, holidays_df=None
                    )
                    st.session_state.prophet_forecast_df_renamed = renamed_fc_df
                    st.session_state.raw_prophet_forecast_for_plotting = raw_fc_obj
                    st.session_state.prophet_metrics = mets
                    st.session_state.prophet_model_fitted = model_fit
                    st.success("Prophet forecast generated!")
                except Exception as e:
                    st.error(f"Error during Prophet forecasting: {e}")
                    for key_to_del in ['prophet_forecast_df_renamed', 'raw_prophet_forecast_for_plotting', 
                                'prophet_metrics', 'prophet_model_fitted', 'final_future_features_df_prophet']:
                        if key_to_del in st.session_state: del st.session_state[key_to_del]
        
        if 'prophet_model_fitted' in st.session_state and \
           'raw_prophet_forecast_for_plotting' in st.session_state and \
           'prophet_forecast_df_renamed' in st.session_state:
            
            prophet_model_to_plot = st.session_state.prophet_model_fitted
            df_for_prophet_native_plot = st.session_state.raw_prophet_forecast_for_plotting
            forecast_display_df_renamed = st.session_state.prophet_forecast_df_renamed
            metrics_fc_disp = st.session_state.prophet_metrics

            st.markdown("---"); st.subheader("Prophet Forecast Results for Lindt Chocolate") # Updated
            hist_plot_fc_disp = historical_df_for_forecast.copy()
            if 'date' not in hist_plot_fc_disp.columns: hist_plot_fc_disp = hist_plot_fc_disp.reset_index()
            
            fig_fc_plot_disp = plot_plotly(prophet_model_to_plot, df_for_prophet_native_plot) 
            fig_fc_plot_disp.add_trace(go.Scatter(x=pd.to_datetime(hist_plot_fc_disp['date']), y=hist_plot_fc_disp['demand'], mode='lines', name='Historical Lindt Demand', line=dict(color='blue'))) # Updated
            fig_fc_plot_disp.update_layout(title='Lindt Chocolate Demand & Prophet Forecast', xaxis_title='Date', yaxis_title='Demand (Units)'); st.plotly_chart(fig_fc_plot_disp, use_container_width=True) # Updated
            
            col_met1, col_met2 = st.columns(2)
            with col_met1: st.metric(label="Forecast RMSE (In-sample Fit)", value=f"{metrics_fc_disp.get('rmse', 'N/A'):.2f}" if pd.notna(metrics_fc_disp.get('rmse')) else "N/A")
            with col_met2: st.metric(label="Forecast MAPE (In-sample Fit)", value=f"{metrics_fc_disp.get('mape', 'N/A'):.1f}%" if pd.notna(metrics_fc_disp.get('mape')) else "N/A")
            
            with st.expander("View Prophet Model Components", expanded=False):
                try: fig_comp_plot_disp = plot_components_plotly(prophet_model_to_plot, df_for_prophet_native_plot); st.plotly_chart(fig_comp_plot_disp, use_container_width=True)
                except Exception as e: st.warning(f"Could not plot Prophet components: {e}")
            
            with st.expander("View User-Defined Future Features Used", expanded=False):
                if 'final_future_features_df_prophet' in st.session_state:
                    st.dataframe(st.session_state.final_future_features_df_prophet)
            
            st.session_state.demand_series_for_simulation = forecast_display_df_renamed['forecasted_demand'].values.tolist()
        elif st.session_state.get('historical_df') is not None:
             st.info("Define future feature plans above and click 'Generate Demand Forecast for Lindt Chocolate'.") # Updated

# --- Tab 4: Inventory Simulation and Optimization ---
with tab_simulate:
    st.markdown("# ⚙️ Inventory Simulation & Optimization") # Updated

    if 'demand_series_for_simulation' not in st.session_state or not st.session_state.demand_series_for_simulation:
        st.warning("Please generate a forecast on the '🔮 Demand Forecasting' tab first.")
    else:
        forecasted_demand_sim_base = st.session_state.demand_series_for_simulation
        st.info(f"Using forecasted Lindt chocolate demand for {len(forecasted_demand_sim_base)} days as baseline.") # Updated

        st.subheader("Simulation Setup") # Updated
        main_controls_col1, main_controls_col2 = st.columns(2)

        with main_controls_col1:
            st.markdown("##### Core Simulation Parameters")
            forecast_len_sim_ctrl = len(forecasted_demand_sim_base)
            sim_duration_sim_ctrl = st.slider("Simulation Duration (Days)", 30, forecast_len_sim_ctrl, min(forecast_len_sim_ctrl, 365), 30, key="sim_duration_main")
            random_seed_sim_ctrl = st.number_input("Random Seed", value=42, step=1, key="rand_seed_main")
            initial_inventory_sim_ctrl = st.number_input("Initial Lindt Stock (Units)", min_value=0, value=500, step=10, key="init_inv_main") # Updated example value
            
            st.markdown("##### Supplier Lead Time for Lindt") # Updated
            mean_lt_sim_ctrl = st.number_input("Mean Lead Time (days)", min_value=1, value=10, step=1, key="mean_lt_main") # Updated example value
            std_lt_sim_ctrl = st.number_input("Std Dev Lead Time (days)", min_value=0, value=3, step=1, key="std_lt_main") # Updated example value

        with main_controls_col2:
            st.markdown("##### Lindt Chocolate Costs") # Updated
            item_cost_sim_ctrl = st.number_input("Cost per Lindt Bar ($)", min_value=0.01, value=2.50, step=0.10, format="%.2f", key="item_cost_main") # Updated example
            holding_rate_sim_ctrl = st.slider("Daily Holding Cost Rate (% item cost)", 0.0, 3.0, 0.5, 0.05, format="%.2f%%", key="hold_rate_main") # Updated example, % per day
            ordering_cost_sim_ctrl = st.number_input("Ordering Cost ($/order)", min_value=0.0, value=75.0, step=5.0, format="%.2f", key="order_cost_main") # Updated example
            stockout_pen_sim_ctrl = st.number_input("Stockout Penalty ($/bar short)", min_value=0.0, value=5.00, step=0.50, format="%.2f", key="stockout_main") # Updated example

            st.markdown("##### Overall Scenario Adjustment (on Forecasted Lindt Demand)") # Updated
            overall_multiplier_sim_ctrl = st.slider("Forecast Multiplier", 0.5, 2.0, 1.0, 0.05, format="%.2fx", key="overall_multi_main")

        st.markdown("---")
        st.subheader("Inventory Policy for Lindt Chocolate: (s, S)") # Updated
        policy_col1, policy_col2 = st.columns(2)
        with policy_col1: s_sim_ctrl = st.number_input("Reorder Point (s) - Min Stock", 0, value=200, step=10, key="s_main") # Updated example
        with policy_col2: S_sim_ctrl = st.number_input("Order-Up-To-Level (S) - Target Stock", s_sim_ctrl + 1, value=max(s_sim_ctrl +1, 600), step=10, key="S_main") # Updated example
        if S_sim_ctrl <= s_sim_ctrl: st.warning("S should be > s.")

        if st.button("🚀 Run Lindt Chocolate Inventory Simulation", key="run_sim_btn_main", type="primary"): # Updated
            sim_demand_base_run_ctrl = forecasted_demand_sim_base[:sim_duration_sim_ctrl]
            sim_demand_final_run_ctrl = [max(0, round(d * overall_multiplier_sim_ctrl)) for d in sim_demand_base_run_ctrl]
            st.session_state.final_demand_for_sim_display = sim_demand_final_run_ctrl

            sim_params_run_ctrl = {
                "random_seed": random_seed_sim_ctrl, "initial_inventory": initial_inventory_sim_ctrl,
                "item_cost": item_cost_sim_ctrl, "holding_cost_rate": holding_rate_sim_ctrl / 100.0,
                "ordering_cost": ordering_cost_sim_ctrl, "stockout_penalty_per_unit": stockout_pen_sim_ctrl,
                "mean_lead_time": mean_lt_sim_ctrl, "std_dev_lead_time": std_lt_sim_ctrl,
                "reorder_point": s_sim_ctrl, "order_up_to_level": S_sim_ctrl,
            }
            with st.spinner(f"Running simulation for Lindt chocolate..."): # Updated
                sim_results_run = run_simulation(sim_params_run_ctrl, sim_demand_final_run_ctrl)
                st.session_state.simulation_results = sim_results_run
        
        if 'simulation_results' in st.session_state and st.session_state.simulation_results:
            results_sim_disp = st.session_state.simulation_results
            st.markdown("---"); st.header("📊 Lindt Chocolate Simulation Results") # Updated
            st.subheader(f"Results for {results_sim_disp['simulated_days']} Days")
            
            # KPIs Display
            st.subheader("Key Performance Indicators (KPIs)")
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Total Operational Cost", f"${results_sim_disp['total_operational_cost']:.2f}")
            kpi_cols[0].metric("Avg. Lindt Stock Level", f"{results_sim_disp['average_inventory_level']:.2f} units") # Updated
            kpi_cols[1].metric("Service Level", f"{results_sim_disp['service_level_percentage']:.2f}%")
            kpi_cols[1].metric("Stockout Incidents", f"{results_sim_disp['num_stockout_incidents']}")
            kpi_cols[2].metric("Orders Placed", f"{results_sim_disp['num_orders_placed']}")
            kpi_cols[2].metric("Total Stockout Units (Lindt Bars)", f"{results_sim_disp['total_stockout_units']}") # Updated
            
            st.markdown("---"); st.subheader("Cost Breakdown")
            cost_cols_disp = st.columns(4)
            cost_cols_disp[0].metric("Holding Cost", f"${results_sim_disp['total_holding_cost']:.2f}")
            cost_cols_disp[1].metric("Ordering Cost", f"${results_sim_disp['total_ordering_cost']:.2f}")
            cost_cols_disp[2].metric("Stockout Cost", f"${results_sim_disp['total_stockout_cost']:.2f}")
            cost_cols_disp[3].metric("Purchase Cost (Lindt Bars)", f"${results_sim_disp['total_purchase_cost']:.2f}") # Updated
            
            st.markdown("---"); st.subheader("📈 Lindt Chocolate Inventory & Demand Profiles") # Updated
            ts_plot_sim_disp = results_sim_disp["timestamps"]
            
            fig_sim_inv_plot_disp = go.Figure()
            if len(ts_plot_sim_disp) > 0 and len(results_sim_disp["inventory_levels"]) == len(ts_plot_sim_disp) :
                fig_sim_inv_plot_disp.add_trace(go.Scatter(x=ts_plot_sim_disp, y=results_sim_disp["inventory_levels"], name='Lindt Stock Level', line=dict(color='dodgerblue'))) # Updated
                fig_sim_inv_plot_disp.add_shape(type="line", x0=ts_plot_sim_disp[0], y0=s_sim_ctrl, x1=ts_plot_sim_disp[-1], y1=s_sim_ctrl, line=dict(color="Red",dash="dash"), name=f'Reorder Point (s={s_sim_ctrl})')
                fig_sim_inv_plot_disp.add_shape(type="line", x0=ts_plot_sim_disp[0], y0=S_sim_ctrl, x1=ts_plot_sim_disp[-1], y1=S_sim_ctrl, line=dict(color="Green",dash="dash"), name=f'Order-Up-To (S={S_sim_ctrl})')
            fig_sim_inv_plot_disp.update_layout(title="Lindt Chocolate Stock Level Over Time", yaxis_title="Units"); st.plotly_chart(fig_sim_inv_plot_disp, use_container_width=True) # Updated
            
            fig_sim_dem_plot_disp = go.Figure()
            if len(ts_plot_sim_disp) > 0 and 'final_demand_for_sim_display' in st.session_state and len(st.session_state.final_demand_for_sim_display) >= len(ts_plot_sim_disp):
                 fig_sim_dem_plot_disp.add_trace(go.Scatter(x=ts_plot_sim_disp, y=st.session_state.final_demand_for_sim_display[:len(ts_plot_sim_disp)], name='Demand Used in Sim', line=dict(color='coral')))
            fig_sim_dem_plot_disp.update_layout(title="Lindt Chocolate Demand Profile Used in Simulation", yaxis_title="Units"); st.plotly_chart(fig_sim_dem_plot_disp, use_container_width=True) # Updated
            
            with st.expander("📝 View Detailed Simulation Logs for Lindt Chocolate"): # Updated
                st.subheader("Order Log"); st.dataframe(pd.DataFrame(results_sim_disp["orders_log"])) if results_sim_disp["orders_log"] else st.write("No orders placed.")
                st.subheader("Demand & Stockout Log (Sample)"); st.dataframe(pd.DataFrame(results_sim_disp["demand_log"]).head(100)) if results_sim_disp["demand_log"] else st.write("No demand events.")
            
            st.markdown("---"); st.header("💡 Lindt Policy Parameter Optimization (Sweep)") # Updated
            if st.checkbox("Run (s,S) Parameter Sweep for Lindt (can be slow)", value=False, key="sweep_checkbox_main"): # Updated
                # ... (Sweep UI and logic - remember to use unique keys from the main app version)
                # ... (This part can be copied from the previous app.py but ensure keys are unique like "s_min_sweep_main" etc.)
                # ... (And it should use sim_params_run_ctrl as the base for modifications)
                st.warning("Parameter sweep UI and logic to be fully integrated here using unique keys.")
        elif st.session_state.get('historical_df') is not None and st.session_state.get('demand_series_for_simulation') is not None:
             st.info("Configure simulation parameters and click 'Run Inventory Simulation'.")