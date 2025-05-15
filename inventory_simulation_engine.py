# inventory_simulation_engine.py
import simpy
import numpy as np
import pandas as pd # Keep for potential future use, not strictly needed now
import matplotlib.pyplot as plt # Only for the __main__ test block
import os

# Global variables for simulation data collection per run
# These are reset in run_simulation
inventory_levels_g = []
timestamps_g = []
orders_placed_log_g = []
demand_log_g = [] # (time, actual_demand_after_scenario, satisfied, stockout)
costs_log_g = {}

class InventorySystem:
    def __init__(self, env, initial_inventory, item_cost, holding_cost_rate,
                 ordering_cost, stockout_penalty_per_unit,
                 scenario_modified_demand_series, 
                 mean_lead_time, std_dev_lead_time,
                 reorder_point, order_up_to_level):
        self.env = env
        self.inventory = initial_inventory
        self.inventory_position = initial_inventory 
        self.item_cost = item_cost
        self.holding_cost_per_unit_day = item_cost * holding_cost_rate
        self.ordering_cost = ordering_cost
        self.stockout_penalty_per_unit = stockout_penalty_per_unit

        self.demand_series = scenario_modified_demand_series
        self.current_day_index = 0

        self.mean_lead_time = mean_lead_time
        self.std_dev_lead_time = std_dev_lead_time
        
        self.reorder_point = reorder_point
        self.order_up_to_level = order_up_to_level
        
        self.units_ordered = 0 
        self.num_orders = 0
        self.num_stockouts_incidents = 0 # Count of times a stockout event occurred
        self.total_stockout_units = 0
        self.total_demand_units = 0
        self.total_satisfied_demand_units = 0

        # Start processes
        self.env.process(self.demand_process())
        self.env.process(self.inventory_control_process())
        self.env.process(self.calculate_daily_holding_cost())

    def place_order(self, quantity):
        global orders_placed_log_g, costs_log_g # Use global for this run
        
        self.num_orders += 1
        costs_log_g["ordering_cost"] += self.ordering_cost
        costs_log_g["purchase_cost"] += quantity * self.item_cost
        self.units_ordered += quantity
        self.inventory_position += quantity 
        
        orders_placed_log_g.append({'time': self.env.now, 'quantity': quantity, 'event': "placed"})
        # print(f"{self.env.now:.2f}: Placed order for {quantity} units. Inv Pos: {self.inventory_position}")
        
        lead_time = max(1, int(round(np.random.normal(self.mean_lead_time, self.std_dev_lead_time))))
        yield self.env.timeout(lead_time) 
        
        self.inventory += quantity
        self.units_ordered -= quantity
        # Inventory position doesn't change here, it was updated at order placement
        # It will implicitly correct itself when inventory_control_process recalculates based on new on-hand.
        orders_placed_log_g.append({'time': self.env.now, 'quantity': quantity, 'event': "arrived"})
        # print(f"{self.env.now:.2f}: Order for {quantity} units arrived. On-hand: {self.inventory}")

    def demand_process(self):
        global demand_log_g, timestamps_g, inventory_levels_g, costs_log_g # Use global for this run

        while self.current_day_index < len(self.demand_series):
            demand_qty = self.demand_series[self.current_day_index]
            self.current_day_index += 1
            self.total_demand_units += demand_qty
            
            actual_demand_this_step = demand_qty 
            satisfied_qty = 0
            stockout_units_this_step = 0

            if demand_qty > 0:
                if self.inventory >= demand_qty:
                    self.inventory -= demand_qty
                    self.inventory_position -= demand_qty 
                    satisfied_qty = demand_qty
                else: # Stockout
                    stockout_units_this_step = demand_qty - self.inventory
                    satisfied_qty = self.inventory
                    self.inventory = 0 # Depleted
                    self.inventory_position -= satisfied_qty # Inv position only drops by what was met
                    
                    self.num_stockouts_incidents += 1
                    self.total_stockout_units += stockout_units_this_step
                    costs_log_g["stockout_cost"] += stockout_units_this_step * self.stockout_penalty_per_unit
            
            self.total_satisfied_demand_units += satisfied_qty
            demand_log_g.append({
                'time': self.env.now, 
                'demanded': actual_demand_this_step, 
                'satisfied': satisfied_qty, 
                'stockout': stockout_units_this_step
            })
            
            timestamps_g.append(self.env.now)
            inventory_levels_g.append(self.inventory)
            yield self.env.timeout(1) # New demand occurs/evaluated each day

    def inventory_control_process(self):
        while True:
            current_inventory_position = self.inventory + self.units_ordered

            if current_inventory_position <= self.reorder_point:
                quantity_to_order = self.order_up_to_level - current_inventory_position
                if quantity_to_order > 0:
                    self.env.process(self.place_order(quantity_to_order))
            
            yield self.env.timeout(1) # Check inventory policy daily

    def calculate_daily_holding_cost(self):
        global costs_log_g # Use global for this run
        while True:
            yield self.env.timeout(1) # Wait for the end of the day
            if self.inventory > 0 : 
                costs_log_g["holding_cost"] += self.inventory * self.holding_cost_per_unit_day


def run_simulation(sim_params, # Core sim settings like costs, policy, lead times
                   demand_series_for_sim, # This is the forecast (potentially scenario-adjusted)
                   ):
    # Reset global lists for this run
    global inventory_levels_g, timestamps_g, orders_placed_log_g, demand_log_g, costs_log_g
    inventory_levels_g = []
    timestamps_g = []
    orders_placed_log_g = []
    demand_log_g = []
    costs_log_g = {"holding_cost": 0, "ordering_cost": 0, "stockout_cost": 0, "purchase_cost": 0}

    np.random.seed(sim_params.get("random_seed", 42))
    env = simpy.Environment()
    
    actual_sim_days = len(demand_series_for_sim) 
    if actual_sim_days == 0:
        # Handle empty demand series case to prevent errors
        return {
            "timestamps": np.array([]), "inventory_levels": np.array([]),
            "actual_demand_series": [], "orders_log": [], "demand_log": [],
            "total_holding_cost": 0, "total_ordering_cost": 0, "total_stockout_cost": 0,
            "total_purchase_cost": 0, "total_operational_cost": 0,
            "num_orders_placed": 0, "num_stockout_incidents": 0, "total_stockout_units": 0,
            "service_level_percentage": 100, "average_inventory_level": sim_params["initial_inventory"],
            "total_demand_units": 0, "total_satisfied_demand": 0,
            "simulated_days": 0
        }

    inventory_system = InventorySystem(
        env,
        initial_inventory=sim_params["initial_inventory"],
        item_cost=sim_params["item_cost"],
        holding_cost_rate=sim_params["holding_cost_rate"],
        ordering_cost=sim_params["ordering_cost"],
        stockout_penalty_per_unit=sim_params["stockout_penalty_per_unit"],
        scenario_modified_demand_series=demand_series_for_sim,
        mean_lead_time=sim_params["mean_lead_time"],
        std_dev_lead_time=sim_params["std_dev_lead_time"],
        reorder_point=sim_params["reorder_point"],
        order_up_to_level=sim_params["order_up_to_level"]
    )
    
    env.run(until=actual_sim_days)
    
    total_op_cost = costs_log_g["holding_cost"] + costs_log_g["ordering_cost"] + costs_log_g["stockout_cost"]
    
    service_level = 0
    if inventory_system.total_demand_units > 0:
        service_level = (inventory_system.total_satisfied_demand_units / inventory_system.total_demand_units) * 100
    else: # No demand means 100% service level (vacuously true) or undefined, choose 100 for display
        service_level = 100
            
    avg_inventory_level = np.mean(inventory_levels_g) if inventory_levels_g else sim_params["initial_inventory"]
    
    results = {
        "timestamps": np.array(timestamps_g),
        "inventory_levels": np.array(inventory_levels_g),
        "actual_demand_series": demand_series_for_sim, # The demand used for THIS run
        "orders_log": list(orders_placed_log_g), # Ensure it's a new list copy
        "demand_log": list(demand_log_g),       # Ensure it's a new list copy
        "total_holding_cost": costs_log_g["holding_cost"],
        "total_ordering_cost": costs_log_g["ordering_cost"],
        "total_stockout_cost": costs_log_g["stockout_cost"],
        "total_purchase_cost": costs_log_g["purchase_cost"],
        "total_operational_cost": total_op_cost,
        "num_orders_placed": inventory_system.num_orders,
        "num_stockout_incidents": inventory_system.num_stockouts_incidents,
        "total_stockout_units": inventory_system.total_stockout_units,
        "service_level_percentage": service_level,
        "average_inventory_level": avg_inventory_level,
        "total_demand_units": inventory_system.total_demand_units,
        "total_satisfied_demand": inventory_system.total_satisfied_demand_units,
        "simulated_days": actual_sim_days
    }
    return results