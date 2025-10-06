def run_causal(data):
    from dowhy import CausalModel

    # Define the causal model
    model = CausalModel(
        data=data,
        treatment="model_name",
        outcome="toxicity_score",
        graph="digraph { model_name -> toxicity_score; }"
    )

    identified_estimand = model.identify_effect()

    # Estimate effect before intervention
    estimate_before = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    # ✅ Handle std_error safely
    std_error = getattr(estimate_before, "stderr", None)
    if std_error is None:
        std_error = "Not available"

    metrics_before = {
        "effect_estimate": estimate_before.value,
        "std_error": std_error
    }

    # 🧩 Simulate intervention (GPT-2 → GPT-Neo)
    data["model_name"] = data["model_name"].replace({"GPT-2": "GPT-Neo"})

    # Re-estimate after intervention
    estimate_after = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    std_error_after = getattr(estimate_after, "stderr", None)
    if std_error_after is None:
        std_error_after = "Not available"

    metrics_after = {
        "effect_estimate": estimate_after.value,
        "std_error": std_error_after
    }

    return model, metrics_before, metrics_after
