import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def process_emotion_series(emotion_confidences):
    emotion_array = np.array(emotion_confidences)  # shape: (n_frames, 7)

    # Dominant emotion per frame
    dominant_indices = np.argmax(emotion_array, axis=1)

    # 1. Emotional Volatility (normalized std of dominant indices)
    volatility = np.std(dominant_indices) / 6  # max spread is 0–6

    # 2. Expression Change Count (normalized)
    expression_changes = np.sum(dominant_indices[:-1] != dominant_indices[1:])
    expression_change_count = expression_changes / (len(dominant_indices) - 1)

    # 3. Microexpression Count (spikes in non-dominant emotions)
    micro_count = 0
    threshold = 0.3
    for i in range(1, len(emotion_array)):
        prev = emotion_array[i - 1]
        curr = emotion_array[i]
        diffs = curr - prev
        # Count only significant upward spikes in non-dominants
        for j in range(7):
            if j != np.argmax(curr) and diffs[j] > threshold:
                micro_count += 1

    microexpression_count = micro_count / ((len(emotion_array) - 1) * 6)  # Normalize to 0–1

    return volatility, microexpression_count, expression_change_count

def define_five_mfs(var):
    var['very_low'] = fuzz.trimf(var.universe, [0.0, 0.0, 0.2])
    var['low'] = fuzz.trimf(var.universe, [0.1, 0.25, 0.4])
    var['medium'] = fuzz.trimf(var.universe, [0.35, 0.5, 0.65])
    var['high'] = fuzz.trimf(var.universe, [0.6, 0.75, 0.9])
    var['very_high'] = fuzz.trimf(var.universe, [0.8, 1.0, 1.0])



def emotion_stablity(input_state):
    
    emotion_volatility = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'emotion_volatility')
    microexpression_count = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'microexpression_count')
    expression_change_count = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'expression_change_count')

    
    emotional_stability = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'emotional_stability')
    
    for var in [emotion_volatility, microexpression_count, expression_change_count, emotional_stability]:
        define_five_mfs(var)
    
    rules_em = [
    # Very stable cases
    ctrl.Rule(emotion_volatility['very_low'] & expression_change_count['very_low'] & microexpression_count['very_low'], emotional_stability['very_high']),
    ctrl.Rule(emotion_volatility['low'] & expression_change_count['low'] & microexpression_count['low'], emotional_stability['high']),

    # Moderately stable
    ctrl.Rule(emotion_volatility['medium'] & expression_change_count['medium'], emotional_stability['medium']),
    ctrl.Rule(emotion_volatility['low'] & expression_change_count['medium'], emotional_stability['medium']),
    ctrl.Rule(emotion_volatility['medium'] & expression_change_count['low'], emotional_stability['medium']),

    # Unstable cases
    ctrl.Rule(emotion_volatility['high'] & expression_change_count['high'], emotional_stability['low']),
    ctrl.Rule(emotion_volatility['very_high'] | expression_change_count['very_high'] | microexpression_count['very_high'], emotional_stability['very_low']),

    # Fallbacks
    ctrl.Rule(emotion_volatility['very_low'] | expression_change_count['very_low'], emotional_stability['high']),
    ctrl.Rule(emotion_volatility['low'] | microexpression_count['low'], emotional_stability['medium']),
    ctrl.Rule(emotion_volatility['medium'] | expression_change_count['medium'] | microexpression_count['medium'], emotional_stability['medium']),
    ctrl.Rule(emotion_volatility['high'] | microexpression_count['high'], emotional_stability['low']),
    ctrl.Rule(emotion_volatility['very_high'] & microexpression_count['very_high'], emotional_stability['very_low']),
    ]


    emotion_confidences=[]
    em_vol,mic_exp,exp_cha_cou=process_emotion_series(input_state)
    em_stability_ctrl = ctrl.ControlSystem(rules_em)
    em_stability_simulator = ctrl.ControlSystemSimulation(em_stability_ctrl)
    em_stability_simulator.input['emotion_volatility'] = em_vol
    em_stability_simulator.input['microexpression_count'] = mic_exp
    em_stability_simulator.input['expression_change_count'] = exp_cha_cou

    
    em_stability_simulator.compute()
    stabilty_value = round(em_stability_simulator.output['emotional_stability'],3)
    return stabilty_value