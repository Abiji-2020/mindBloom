import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_pause_frequency(input_reaction_time, input_speed, input_range, input_symmetry):
    reaction_time = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'reaction_time')
    speed = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'speed')
    range_ = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'range')
    symmetry = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'symmetry')

# Define fuzzy output variable
    pause_frequency = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'pause_frequency')

# Membership functions for inputs
    for var in [reaction_time, speed, range_, symmetry]:
        var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
        var['medium'] = fuzz.trimf(var.universe, [0.25, 0.5, 0.75])
        var['high'] = fuzz.trimf(var.universe, [0.5, 1, 1])

# Membership functions for output
    pause_frequency['very_low'] = fuzz.trimf(pause_frequency.universe, [0, 0, 0.2])
    pause_frequency['low'] = fuzz.trimf(pause_frequency.universe, [0.1, 0.3, 0.5])
    pause_frequency['medium'] = fuzz.trimf(pause_frequency.universe, [0.4, 0.5, 0.6])
    pause_frequency['high'] = fuzz.trimf(pause_frequency.universe, [0.5, 0.7, 0.9])
    pause_frequency['very_high'] = fuzz.trimf(pause_frequency.universe, [0.8, 1, 1])

# Define fuzzy rules (25 rules)
    rules = [
    ctrl.Rule(reaction_time['low'] & speed['high'] & range_['high'] & symmetry['high'], pause_frequency['very_low']),
    ctrl.Rule(reaction_time['medium'] & speed['medium'] & range_['medium'] & symmetry['medium'], pause_frequency['medium']),
    ctrl.Rule(reaction_time['high'] & speed['low'] & range_['low'] & symmetry['low'], pause_frequency['very_high']),

    ctrl.Rule(reaction_time['low'] & speed['low'], pause_frequency['high']),
    ctrl.Rule(speed['low'] & symmetry['low'], pause_frequency['high']),
    ctrl.Rule(range_['low'] & symmetry['high'], pause_frequency['medium']),
    ctrl.Rule(speed['medium'] & range_['high'], pause_frequency['low']),
    ctrl.Rule(reaction_time['high'] & speed['medium'], pause_frequency['medium']),
    ctrl.Rule(speed['high'] & symmetry['medium'], pause_frequency['low']),
    ctrl.Rule(range_['medium'] & symmetry['medium'], pause_frequency['medium']),

    ctrl.Rule(reaction_time['medium'] & speed['high'], pause_frequency['low']),
    ctrl.Rule(reaction_time['low'] & symmetry['medium'], pause_frequency['low']),
    ctrl.Rule(speed['low'] & range_['medium'], pause_frequency['medium']),
    ctrl.Rule(speed['high'] & range_['low'], pause_frequency['medium']),
    ctrl.Rule(range_['high'] & symmetry['low'], pause_frequency['high']),

    ctrl.Rule(reaction_time['low'] & range_['high'], pause_frequency['low']),
    ctrl.Rule(reaction_time['medium'] & symmetry['low'], pause_frequency['medium']),
    ctrl.Rule(reaction_time['high'] & symmetry['high'], pause_frequency['low']),
    ctrl.Rule(reaction_time['medium'] & range_['low'], pause_frequency['high']),
    ctrl.Rule(reaction_time['high'] & range_['medium'], pause_frequency['medium']),

    ctrl.Rule(speed['low'] & symmetry['medium'], pause_frequency['medium']),
    ctrl.Rule(speed['medium'] & symmetry['high'], pause_frequency['low']),
    ctrl.Rule(range_['low'] & speed['high'], pause_frequency['low']),
    ctrl.Rule(reaction_time['low'] & speed['medium'] & symmetry['low'], pause_frequency['medium']),
    ctrl.Rule(reaction_time['high'] & speed['low'] & symmetry['medium'], pause_frequency['high']),
    ]

# Create control system and simulation
    pause_ctrl = ctrl.ControlSystem(rules)
    pause_simulation = ctrl.ControlSystemSimulation(pause_ctrl)

# Example input for prediction
    pause_simulation.input['reaction_time'] = input_reaction_time
    pause_simulation.input['speed'] = input_speed
    pause_simulation.input['range'] = input_range
    pause_simulation.input['symmetry'] = input_symmetry

    pause_simulation.compute()
    pas_freq = round(pause_simulation.output['pause_frequency'],3)
    return pas_freq