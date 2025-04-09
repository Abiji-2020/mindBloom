import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# speed, range, symmetrry
def get_reaction_time_ms(data): 
    speed = ctrl.Antecedent(np.linspace(0, 1, 100), 'speed')
    range_ = ctrl.Antecedent(np.linspace(0, 1, 100), 'range')
    symmetry = ctrl.Antecedent(np.linspace(0, 1, 100), 'symmetry')

# Define output variable
    reaction_time = ctrl.Consequent(np.linspace(0, 1, 100), 'normalized_reaction_time')

# Membership functions for inputs (Low, Medium, High)
    for var in [speed, range_, symmetry]:
        var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
        var['medium'] = fuzz.trimf(var.universe, [0.25, 0.5, 0.75])
        var['high'] = fuzz.trimf(var.universe, [0.5, 1.0, 1.0])

# Membership functions for output (Very High = slowest, Very Low = fastest)
    reaction_time['very_low'] = fuzz.trimf(reaction_time.universe, [0, 0, 0.2])
    reaction_time['low'] = fuzz.trimf(reaction_time.universe, [0.1, 0.25, 0.4])
    reaction_time['medium'] = fuzz.trimf(reaction_time.universe, [0.3, 0.5, 0.7])
    reaction_time['high'] = fuzz.trimf(reaction_time.universe, [0.6, 0.75, 0.9])
    reaction_time['very_high'] = fuzz.trimf(reaction_time.universe, [0.8, 1.0, 1.0])

# Define fuzzy rules (20 meaningful rules)
    rules = [
    ctrl.Rule(speed['high'] & range_['high'] & symmetry['high'], reaction_time['very_low']),
    ctrl.Rule(speed['low'] | range_['low'], reaction_time['very_high']),
    ctrl.Rule(speed['medium'] & range_['medium'] & symmetry['medium'], reaction_time['medium']),
    ctrl.Rule(speed['high'] & symmetry['low'], reaction_time['medium']),
    ctrl.Rule(speed['low'] & range_['high'], reaction_time['high']),
    ctrl.Rule(speed['medium'] & range_['low'], reaction_time['high']),
    ctrl.Rule(speed['low'] & symmetry['low'], reaction_time['very_high']),
    ctrl.Rule(speed['high'] & range_['medium'] & symmetry['high'], reaction_time['low']),
    ctrl.Rule(speed['medium'] & range_['high'], reaction_time['low']),
    ctrl.Rule(speed['medium'] & symmetry['low'], reaction_time['medium']),
    ctrl.Rule(range_['medium'] & symmetry['medium'], reaction_time['medium']),
    ctrl.Rule(speed['low'] & range_['medium'] & symmetry['medium'], reaction_time['high']),
    ctrl.Rule(speed['high'] & range_['low'] & symmetry['low'], reaction_time['high']),
    ctrl.Rule(speed['medium'] & range_['medium'] & symmetry['low'], reaction_time['medium']),
    ctrl.Rule(speed['high'] & range_['high'] & symmetry['low'], reaction_time['low']),
    ctrl.Rule(speed['low'] & range_['low'] & symmetry['high'], reaction_time['very_high']),
    ctrl.Rule(speed['low'] & range_['high'] & symmetry['low'], reaction_time['high']),
    ctrl.Rule(speed['high'] & range_['low'] & symmetry['high'], reaction_time['medium']),
    ctrl.Rule(speed['medium'] & range_['low'] & symmetry['high'], reaction_time['medium']),
    ctrl.Rule(speed['medium'] & range_['high'] & symmetry['low'], reaction_time['medium'])
    ]

# Build control system
    reaction_ctrl = ctrl.ControlSystem(rules)
    reaction_simulator = ctrl.ControlSystemSimulation(reaction_ctrl)

# Function to process an array of [speed, range, symmetry] inputs
    def compute_reaction_times(data_matrix):
        results = []
        for row in data_matrix:
            s, r, sym = row
            reaction_simulator.input['speed'] = s
            reaction_simulator.input['range'] = r
            reaction_simulator.input['symmetry'] = sym
            reaction_simulator.compute()
            results.append(reaction_simulator.output['normalized_reaction_time'])
        return np.array(results)
    result = compute_reaction_times(data)
    return result

