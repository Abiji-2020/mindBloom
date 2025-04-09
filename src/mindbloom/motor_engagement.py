import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mindbloom.reaction_time_ms import get_reaction_time_ms

def min_max_normalize(arr, method='mean'):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized = (arr - min_val) / (max_val - min_val + 1e-8)  # Avoid div-by-zero

    if method == 'mean':
        return np.mean(normalized)
    elif method == 'sum':
        return np.sum(normalized)
    elif method == 'weighted':
        weights = np.linspace(1, 2, len(arr))  # Custom increasing weights
        return np.dot(normalized, weights) / np.sum(weights)
    else:
        raise ValueError("Method must be 'mean', 'sum', or 'weighted'.")


def get_mortor_engagement(input_speed, input_range, input_symmetry):
    input_reaction_time = get_reaction_time_ms(np.array(list(zip(input_speed, input_range, input_symmetry))))
    reaction_time = ctrl.Antecedent(np.linspace(0, 1, 100), 'reaction_time')
    speed = ctrl.Antecedent(np.linspace(0, 1, 100), 'speed')
    range_ = ctrl.Antecedent(np.linspace(0, 1, 100), 'range')

# Define output
    motor_engagement = ctrl.Consequent(np.linspace(0, 1, 100), 'motor_engagement')

# Membership functions for inputs (Low, Medium, High)
    for var in [reaction_time, speed, range_]:
        var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
        var['medium'] = fuzz.trimf(var.universe, [0.25, 0.5, 0.75])
        var['high'] = fuzz.trimf(var.universe, [0.5, 1.0, 1.0])

# Membership functions for reaction_time (Very Low to Very High)
    reaction_time['very_low'] = fuzz.trimf(reaction_time.universe, [0, 0, 0.2])
    reaction_time['low'] = fuzz.trimf(reaction_time.universe, [0.1, 0.25, 0.4])
    reaction_time['medium'] = fuzz.trimf(reaction_time.universe, [0.3, 0.5, 0.7])
    reaction_time['high'] = fuzz.trimf(reaction_time.universe, [0.6, 0.75, 0.9])
    reaction_time['very_high'] = fuzz.trimf(reaction_time.universe, [0.8, 1.0, 1.0])

# Membership functions for output
    motor_engagement['very_low'] = fuzz.trimf(motor_engagement.universe, [0, 0, 0.2])
    motor_engagement['low'] = fuzz.trimf(motor_engagement.universe, [0.1, 0.25, 0.4])
    motor_engagement['medium'] = fuzz.trimf(motor_engagement.universe, [0.3, 0.5, 0.7])
    motor_engagement['high'] = fuzz.trimf(motor_engagement.universe, [0.6, 0.75, 0.9])
    motor_engagement['very_high'] = fuzz.trimf(motor_engagement.universe, [0.8, 1.0, 1.0])

# Define 25 fuzzy rules
    rules = [
        ctrl.Rule(reaction_time['very_low'] & speed['high'] & range_['high'], motor_engagement['very_high']),
        ctrl.Rule(reaction_time['very_high'], motor_engagement['very_low']),
        ctrl.Rule(reaction_time['medium'] & speed['medium'] & range_['medium'], motor_engagement['medium']),
        ctrl.Rule(speed['low'] | range_['low'], motor_engagement['low']),
        ctrl.Rule(speed['high'] & range_['medium'], motor_engagement['high']),
        ctrl.Rule(speed['medium'] & range_['high'], motor_engagement['high']),
        ctrl.Rule(speed['medium'] & range_['low'], motor_engagement['low']),
        ctrl.Rule(reaction_time['low'] & speed['medium'] & range_['medium'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['low'] & speed['high'], motor_engagement['high']),
        ctrl.Rule(reaction_time['high'] & speed['low'], motor_engagement['low']),
        ctrl.Rule(reaction_time['very_low'] & range_['high'], motor_engagement['very_high']),
        ctrl.Rule(reaction_time['very_high'] & speed['medium'], motor_engagement['low']),
        ctrl.Rule(reaction_time['very_low'] & speed['low'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['low'] & range_['low'], motor_engagement['low']),
        ctrl.Rule(reaction_time['medium'] & range_['low'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['high'] & speed['high'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['medium'] & speed['low'] & range_['high'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['high'] & speed['medium'] & range_['medium'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['high'] & speed['high'] & range_['low'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['very_low'] & speed['high'] & range_['low'], motor_engagement['high']),
        ctrl.Rule(reaction_time['very_low'] & speed['medium'] & range_['medium'], motor_engagement['high']),
        ctrl.Rule(reaction_time['low'] & speed['low'] & range_['medium'], motor_engagement['low']),
        ctrl.Rule(reaction_time['low'] & speed['high'] & range_['low'], motor_engagement['medium']),
        ctrl.Rule(reaction_time['medium'] & speed['medium'] & range_['high'], motor_engagement['high']),
        ctrl.Rule(reaction_time['very_high'] & speed['high'] & range_['high'], motor_engagement['medium'])
    ]

# Build and simulate control system
    motor_ctrl = ctrl.ControlSystem(rules)
    motor_simulator = ctrl.ControlSystemSimulation(motor_ctrl)

# Function to compute motor engagement values
    def compute_motor_engagement(data_matrix):
        results = []
        for row in data_matrix:
            rt, s, r = row
            motor_simulator.input['reaction_time'] = rt
            motor_simulator.input['speed'] = s
            motor_simulator.input['range'] = r
            motor_simulator.compute()
            results.append(motor_simulator.output['motor_engagement'])
        return np.array(results)

    engagement_score = compute_motor_engagement(data_matrix = np.array(list(zip(input_reaction_time, input_speed, input_range))))
    return round(min_max_normalize(engagement_score), 3)
