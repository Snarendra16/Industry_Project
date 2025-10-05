def run_causal(data):
    # --- Preprocess ---
    data = data.copy()
    data['treatment'] = data['treatment'].astype(str)
    data['outcome'] = data['outcome'].astype(float)

    from dowhy import CausalModel

    model = CausalModel(
        data=data,
        treatment='treatment',
        outcome='outcome',
        common_causes=['gender', 'topic']  # example variables
    )
    
    identified_estimand = model.identify_effect()
    estimate_before = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    
    metrics_before = {
        "effect_estimate": estimate_before.value,
        "std_error": estimate_before.get_std_error()
    }

    # Simulate intervention
    data['treatment'] = data['treatment'].replace({'GPT-2': 'GPT-Neo'})
    estimate_after = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    metrics_after = {
        "effect_estimate": estimate_after.value,
        "std_error": estimate_after.get_std_error()
    }

    return model, metrics_before, metrics_after
