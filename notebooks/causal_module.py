# causal_module.py
import dowhy
from dowhy import CausalModel
import pandas as pd

def run_causal(data):
    """
    Runs causal analysis and returns model and metrics before/after intervention.
    """
    # Example: Assume 'treatment' affects 'outcome'
    model = CausalModel(
        data=data,
        treatment='treatment',
        outcome='outcome',
        common_causes=['age', 'gender', 'income']  # adjust based on your dataset
    )

    # Identify causal effect
    identified_estimand = model.identify_effect()
    
    # Estimate effect using backdoor adjustment
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression")
    
    # Metrics before/after interventions (example: bias reduction)
    metrics_before = {
        "effect_estimate": estimate.value,  # naive/observed estimate
        "std_error": estimate.get_std_error()
    }
    
    # Apply an "intervention" (simulate a treatment)
    # Here we just do a do-operation as an example
    new_estimate = model.do(x=1).estimate_effect(identified_estimand,
                                                method_name="backdoor.linear_regression")
    
    metrics_after = {
        "effect_estimate": new_estimate.value,
        "std_error": new_estimate.get_std_error()
    }

    return model, metrics_before, metrics_after
