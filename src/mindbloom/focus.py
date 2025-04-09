import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mindbloom.pause_frequency import get_pause_frequency
from mindbloom.reaction_time_ms import get_reaction_time_ms

def define_five_mfs(var):
    var['very_low'] = fuzz.trimf(var.universe, [0.0, 0.0, 0.2])
    var['low'] = fuzz.trimf(var.universe, [0.1, 0.25, 0.4])
    var['medium'] = fuzz.trimf(var.universe, [0.35, 0.5, 0.65])
    var['high'] = fuzz.trimf(var.universe, [0.6, 0.75, 0.9])
    var['very_high'] = fuzz.trimf(var.universe, [0.8, 1.0, 1.0])

def get_input_focus(emotion_series):
    num_frames = len(emotion_series)
    dominant_indices = [np.argmax(frame) for frame in emotion_series]
    dominant_values = [max(frame) for frame in emotion_series]

    focus_consistency = sum(dominant_indices[i] == dominant_indices[i+1] for i in range(num_frames - 1)) / (num_frames - 1)
    avg_dominant_strength = np.mean(dominant_values)
    focus = 0.5 * focus_consistency + 0.5 * avg_dominant_strength

    return focus
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

def get_input_speed(speed):
    return min_max_normalize(speed)

def get_input_range(range_):
    return min_max_normalize(range_)

def get_input_symmetry(symmetry):
    return min_max_normalize(symmetry)

def get_reaction_data(speed, range_, symmetry):
    return np.array(list(zip(speed, range_, symmetry)))

def get_focus(input_emotions, input_speed, input_range, input_symmetry):
    focus = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'focus')
    pause_frequency = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'pause_frequency')
    reaction_time = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'reaction_time_ms')

    focus_score = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'focus_attention_score')

    for var in [focus, pause_frequency, reaction_time, focus_score]:
        define_five_mfs(var)

    rules_fa = [
    # Excellent focus behavior
    ctrl.Rule(focus['very_high'] & pause_frequency['very_low'] & reaction_time['very_low'] , focus_score['very_high']),
    ctrl.Rule(focus['high'] & pause_frequency['low'] & reaction_time['low'], focus_score['high']),

    # Good focus, slightly slower response
    ctrl.Rule(focus['high'] & pause_frequency['medium'] & reaction_time['medium'], focus_score['medium']),
    ctrl.Rule(focus['medium'] & pause_frequency['low'] & reaction_time['low'], focus_score['medium']),

    # Mixed focus or higher pauses
    ctrl.Rule(focus['medium'] & pause_frequency['medium'] & reaction_time['medium'], focus_score['medium']),
    ctrl.Rule(focus['low'] & pause_frequency['medium'] & reaction_time['medium'], focus_score['low']),

    # Poor focus and high delays
    ctrl.Rule(focus['low'] & pause_frequency['high'] & reaction_time['high'], focus_score['low']),
    ctrl.Rule(focus['very_low'] & pause_frequency['very_high'], focus_score['very_low']),

    # Worst case: no focus, long completion, many pauses
    ctrl.Rule((focus['very_low'] & pause_frequency['very_high'] & reaction_time['very_high'] ), focus_score['very_low']),

    # Fallbacks
    ctrl.Rule((focus['very_low'] | pause_frequency['very_high'] | reaction_time['very_high'] ), focus_score['very_low']),
    ctrl.Rule((focus['very_high'] | pause_frequency['very_low'] | reaction_time['very_low'] ), focus_score['very_high']),
    # Mixed but recovering behavior
    ctrl.Rule(focus['high'] & reaction_time['high'] , focus_score['medium']),
    ctrl.Rule(focus['medium'] & pause_frequency['low'] & reaction_time['medium'], focus_score['medium']),
    ctrl.Rule(focus['medium'] & pause_frequency['high'] & reaction_time['low'], focus_score['low']),

# Inconsistent but decent overall
    ctrl.Rule(focus['high'] & pause_frequency['high'] & reaction_time['low'], focus_score['medium']),
    ctrl.Rule(focus['medium'] & pause_frequency['very_low'] & reaction_time['very_high'], focus_score['low']),

# Slightly better than average
    ctrl.Rule(focus['medium'] & pause_frequency['medium'] & reaction_time['low'], focus_score['medium']),
    ctrl.Rule(focus['low'] & pause_frequency['low'] & reaction_time['low'], focus_score['medium']),

# Bad recovery (starts well, ends poorly)
    ctrl.Rule(focus['high']  & pause_frequency['medium'], focus_score['low']),
    ctrl.Rule(focus['medium'] & reaction_time['very_high'], focus_score['low']),
    # Weighted fallback based on primary focus
    ctrl.Rule(focus['medium'], focus_score['medium']),
    ctrl.Rule(focus['very_low'] & pause_frequency['medium'], focus_score['low']),
    ctrl.Rule(focus['very_high'] & pause_frequency['medium'], focus_score['high']),
    ctrl.Rule(focus['low'] & pause_frequency['very_low'] & reaction_time['very_low'], focus_score['medium']),
    ctrl.Rule(focus['high'] & pause_frequency['very_low'] & reaction_time['very_low'], focus_score['high']),
    ctrl.Rule(focus['very_high'] & reaction_time['medium'], focus_score['high']),
    ctrl.Rule(focus['medium'] & pause_frequency['medium'] & reaction_time['medium'], focus_score['medium']),
    ctrl.Rule(pause_frequency['very_high'] & reaction_time['very_low'], focus_score['low']),
    ]
    fa_stability_ctrl = ctrl.ControlSystem(rules_fa)
    fa_stability_simulator = ctrl.ControlSystemSimulation(fa_stability_ctrl)
    fa_stability_simulator.input['focus'] = get_input_focus(input_emotions)
    fa_stability_simulator.input['pause_frequency'] = get_pause_frequency(min_max_normalize(get_reaction_time_ms(get_reaction_data(input_speed, input_range, input_symmetry))),get_input_speed(input_speed), get_input_range(input_range), get_input_symmetry(input_symmetry))
    fa_stability_simulator.input['reaction_time_ms'] = min_max_normalize(get_reaction_time_ms(get_reaction_data(input_speed, input_range, input_symmetry)))

    
    fa_stability_simulator.compute()
    focus_score_value = round(fa_stability_simulator.output['focus_attention_score'], 3)
    return focus_score_value

