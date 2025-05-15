# pages/3_⚙️_Inventory_Simulation_and_Optimization.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from inventory_simulation_engine import run_simulation # Assuming this is correctly set up

st.set_page_config(page_title="Inventory Simulation", layout="wide")
st.markdown("# ⚙️ Inventory Simulation & Optimization")
st.sidebar.header("Inventory Simulation")

if 'demand_series_for_simulation' not in st.session_state or not st.session_state.demand_series_for_simulation:
    st.warning("Please generate a forecast on the '🔮 Demand Forecasting' page first.")
    st.stop()

forecasted_demand = st.session_state.demand_series_for_simulation
st.info(f"Using forecasted demand for {len(forecasted_demand)} days as the baseline for simulation.")

# --- Scenario Definition on top of Forecast ---
st.sidebar.subheader("Scenario Adjustment (on Forecast)")
overall_multiplier_app = st.sidebar.slider("Overall Forecast Multiplier", 0.5, 2.0, 1.0, 0.05, format="%.2fx",
                                           help="Adjust the entire forecast up or down (e.g., for optimistic/pessimistic scenarios).")

# --- Core Simulation Parameters ---
st.sidebar.header("⚙️ Core Simulation Parameters")
# sim_time_app is now dictated by forecast length, but we can allow a shorter sim
forecast_length = len(forecasted_demand)
sim_duration_app = st.sidebar.slider("Simulation Duration (Days)", 
                                     min_value=30, 
                                     max_value=forecast_length, 
                                     value=min(forecast_length, 365), # Default to forecast length or 1 year
                                     step=30)
random_seed_app = st.sidebar.number_input("Random Seed", value=42, step=1)

# --- Item & Cost Parameters --- (Copied from previous app version)
st.sidebar.header("🏭 Item & Cost Parameters")
initial_inventory_app = st.sidebar.number_input("Initial Inventory (units)", min_value=0, value=150, step=10)
item_cost_app = st.sidebar.number_input("Item Cost ($/unit)", min_value=0.01, value=10.0, step=0.5, format="%.2f")
holding_cost_rate_app = st.sidebar.slider("Daily Holding Cost Rate (% of item cost)", 0.0, 5.0, 1.0, 0.1, format="%.2f%%") # per day
ordering_cost_app = st.sidebar.number_input("Ordering Cost ($/order)", min_value=0.0, value=50.0, step=5.0, format="%.2f")
stockout_penalty_app = st.sidebar.number_input("Stockout Penalty ($/unit short)", min_value=0.0, value=25.0, step=1.0, format="%.2f")

# --- Lead Time Parameters ---
st.sidebar.header("🚚 Lead Time Parameters")
mean_lead_time_app = st.sidebar.number_input("Mean Lead Time (days)", min_value=1, value=7, step=1)
std_dev_lead_time_app = st.sidebar.number_input("Std Dev of Lead Time (days)", min_value=0, value=2, step=1)

# --- Inventory Policy ---
st.sidebar.header("📜 Inventory Policy: (s, S)")
reorder_point_s_app = st.sidebar.number_input("Reorder Point (s) (units)", min_value=0, value=100, step=5)
order_up_to_S_app = st.sidebar.number_input("Order-Up-To-Level (S) (units)", min_value=1, value=300, step=10)

if order_up_to_S_app <= reorder_point_s_app:
    st.sidebar.error("Order-Up-To-Level (S) must be greater than Reorder Point (s).")
    st.stop()


# --- Simulation Run ---
if st.sidebar.button("🚀 Run Inventory Simulation"):
    # Apply overall multiplier to the forecast for the simulation duration
    sim_demand_series_base = forecasted_demand[:sim_duration_app]
    sim_demand_series_scenario = [max(0, round(d * overall_multiplier_app)) for d in sim_demand_series_base]

    sim_params_dict = {
        "random_seed": random_seed_app,
        "initial_inventory": initial_inventory_app,
        "item_cost": item_cost_app,
        "holding_cost_rate": holding_cost_rate_app / 100.0, # Convert percentage
        "ordering_cost": ordering_cost_app,
        "stockout_penalty_per_unit": stockout_penalty_app,
        "mean_lead_time": mean_lead_time_app,
        "std_dev_lead_time": std_dev_lead_time_app,
        "reorder_point": reorder_point_s_app,
        "order_up_to_level": order_up_to_S_app,
    }

    with st.spinner(f"Running simulation for {sim_duration_app} days..."):
        results = run_simulation(
            sim_params=sim_params_dict,
            demand_series_for_sim=sim_demand_series_scenario # Use the scenario-adjusted forecast
        )
        st.session_state.simulation_results = results

if 'simulation_results' in st.session_state:
    results = st.session_state.simulation_results
    st.markdown("---")
    st.header("📊 Simulation Results")
    st.subheader(f"Results for {results['simulated_days']} Simulated Days (Based on Forecast with Scenario Adjustment)")

    # KPIs Display (same as before)
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Operational Cost", f"${results['total_operational_cost']:.2f}")
        st.metric("Avg. Inventory Level", f"{results['average_inventory_level']:.2f} units")
    with col2:
        st.metric("Service Level", f"{results['service_level_percentage']:.2f}%")
        st.metric("Num. Stockout Incidents", f"{results['num_stockout_incidents']}")
    with col3:
        st.metric("Num. Orders Placed", f"{results['num_orders_placed']}")
        st.metric("Total Stockout Units", f"{results['total_stockout_units']}")

    st.markdown("---")
    st.subheader("Cost Breakdown")
    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
    cost_col1.metric("Total Holding Cost", f"${results['total_holding_cost']:.2f}")
    cost_col2.metric("Total Ordering Cost", f"${results['total_ordering_cost']:.2f}")
    cost_col3.metric("Total Stockout Cost", f"${results['total_stockout_cost']:.2f}")
    cost_col4.metric("Total Purchase Cost", f"${results['total_purchase_cost']:.2f}")
    st.markdown("---")

    # Plotting (Inventory and Demand Profile used)
    st.subheader("📈 Inventory & Demand Profiles")
    fig_sim = go.Figure()
    # Inventory Level
    fig_sim.add_trace(go.Scatter(x=results["timestamps"], y=results["inventory_levels"], mode='lines', name='Inventory Level', line=dict(color='dodgerblue')))
    # Reorder Point and Order-Up-To Level lines
    fig_sim.add_shape(type="line", x0=results["timestamps"][0], y0=reorder_point_s_app, x1=results["timestamps"][-1], y1=reorder_point_s_app, line=dict(color="Red",dash="dash"), name=f'Reorder (s={reorder_point_s_app})')
    fig_sim.add_shape(type="line", x0=results["timestamps"][0], y0=order_up_to_S_app, x1=results["timestamps"][-1], y1=order_up_to_S_app, line=dict(color="Green",dash="dash"), name=f'Order-up-to (S={order_up_to_S_app})')
    
    fig_sim.update_layout(title="Inventory Level Over Time (Under Scenario)", yaxis_title="Inventory Units")
    st.plotly_chart(fig_sim, use_container_width=True)

    fig_demand_used = go.Figure()
    fig_demand_used.add_trace(go.Scatter(x=results["timestamps"], y=results["actual_demand_series"][:len(results["timestamps"])], mode='lines', name='Demand Used in Sim', line=dict(color='coral')))
    fig_demand_used.update_layout(title="Demand Profile Used in Simulation (Forecast + Scenario)", xaxis_title="Simulation Time (Days)", yaxis_title="Demand Units")
    st.plotly_chart(fig_demand_used, use_container_width=True)

    # Logs (Order Log and Demand/Stockout Log)
    with st.expander("📝 View Detailed Simulation Logs"):
        st.subheader("Order Log")
        if results["orders_log"]:
            order_log_df = pd.DataFrame(results["orders_log"])
            order_log_df['time'] = order_log_df['time'].round(1)
            st.dataframe(order_log_df)
        else:
            st.write("No orders were placed during this simulation run.")

        st.subheader("Demand & Stockout Log (Sample of first 100 events)")
        if results["demand_log"]:
            demand_stockout_df = pd.DataFrame(results["demand_log"])
            demand_stockout_df['time'] = demand_stockout_df['time'].round(1)
            st.dataframe(demand_stockout_df.head(100))
        else:
            st.write("No demand events were logged (e.g., simulation duration was 0 or demand series was empty).")
            
    # --- Basic Parameter Sweep for Optimization Insights ---
    st.markdown("---")
    st.header("💡 Policy Parameter Optimization (Sweep)")
    st.markdown("Test a range of (s, S) values to find better performing policies based on the current forecast & scenario.")

    if st.checkbox("Run (s,S) Parameter Sweep (can be slow)", value=False):
        s_min = st.slider("Min 's' for sweep", 0, 200, 50, 10)
        s_max = st.slider("Max 's' for sweep", s_min + 10, 500, 150, 10)
        s_step = st.slider("Step for 's'", 10, 50, 20, 5)
        
        S_min_multiplier = st.slider("Min 'S' (as s + X)", 20, 200, 50, 10) # S should be s + X
        S_max_multiplier = st.slider("Max 'S' (as s + Y)", S_min_multiplier + 20, 500, 200, 10)
        S_step_multiplier = st.slider("Step for 'S' increment", 20, 100, 40, 10)

        if st.button("Start Sweep Simulation"):
            sweep_results = []
            s_values = range(s_min, s_max + 1, s_step)
            
            progress_bar = st.progress(0)
            total_runs = 0
            for s_val_sweep in s_values:
                S_start_val = s_val_sweep + S_min_multiplier
                S_end_val = s_val_sweep + S_max_multiplier
                total_runs += len(range(S_start_val, S_end_val + 1, S_step_multiplier))
            
            current_run = 0

            with st.spinner("Running parameter sweep... Please wait."):
                for s_val_sweep in s_values:
                    S_start_val = s_val_sweep + S_min_multiplier
                    S_end_val = s_val_sweep + S_max_multiplier
                    if S_start_val > S_end_val : S_start_val = S_end_val # ensure S is not less than s + X

                    for S_val_sweep in range(S_start_val, S_end_val + 1, S_step_multiplier):
                        if S_val_sweep <= s_val_sweep: continue # Ensure S > s

                        current_run += 1
                        progress_bar.progress(current_run / total_runs if total_runs > 0 else 0)

                        temp_sim_params = sim_params_dict.copy()
                        temp_sim_params["reorder_point"] = s_val_sweep
                        temp_sim_params["order_up_to_level"] = S_val_sweep
                        
                        # Apply overall multiplier to the forecast for the simulation duration
                        sim_demand_base = forecasted_demand[:sim_duration_app] # Use original forecast length for sweep runs
                        sim_demand_scenario_sweep = [max(0, round(d * overall_multiplier_app)) for d in sim_demand_base]


                        res = run_simulation(temp_sim_params, sim_demand_scenario_sweep)
                        sweep_results.append({
                            's': s_val_sweep,
                            'S': S_val_sweep,
                            'Total Op Cost': res['total_operational_cost'],
                            'Service Level (%)': res['service_level_percentage'],
                            'Avg Inventory': res['average_inventory_level'],
                            'Stockout Units': res['total_stockout_units']
                        })
            
            progress_bar.empty() # Clear progress bar
            if sweep_results:
                sweep_df = pd.DataFrame(sweep_results)
                st.subheader("Parameter Sweep Results")
                st.dataframe(sweep_df.style.format({
                    'Total Op Cost': '${:,.2f}',
                    'Service Level (%)': '{:.2f}%',
                    'Avg Inventory': '{:.2f}',
                }).highlight_min(subset=['Total Op Cost'], color='lightgreen')
                  .highlight_max(subset=['Service Level (%)'], color='lightgreen')
                  .highlight_min(subset=['Stockout Units'], color='lightgreen'))
                
                st.write("Consider policies that balance low cost with high service level.")
                # Simple recommendation
                best_cost_row = sweep_df.loc[sweep_df['Total Op Cost'].idxmin()]
                st.success(f"Lowest Cost Policy Found: s={best_cost_row['s']}, S={best_cost_row['S']} "
                           f"(Cost: ${best_cost_row['Total Op Cost']:.2f}, Service: {best_cost_row['Service Level (%)']:.2f}%)")

                # Filter for high service level policies then find min cost
                high_service_df = sweep_df[sweep_df['Service Level (%)'] >= 98.0] # Example target
                if not high_service_df.empty:
                    best_service_cost_row = high_service_df.loc[high_service_df['Total Op Cost'].idxmin()]
                    st.info(f"Lowest Cost for >=98% Service: s={best_service_cost_row['s']}, S={best_service_cost_row['S']} "
                               f"(Cost: ${best_service_cost_row['Total Op Cost']:.2f}, Service: {best_service_cost_row['Service Level (%)']:.2f}%)")
                else:
                    st.warning("No policies achieved >=98% service level in this sweep.")

            else:
                st.write("No results from parameter sweep (check ranges).")
else:
    st.info("Configure simulation parameters and run the simulation, or optionally run the parameter sweep.")