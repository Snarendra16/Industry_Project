from dowhy import CausalModel

def run_causal(df):
    model = CausalModel(
        data=df,
        treatment="Gender",
        outcome="Sentiment",
        graph="digraph{Gender -> Sentiment; Bias -> Sentiment;}"
    )
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    return estimate.value
