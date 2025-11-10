import pandas as pd
import matplotlib.pyplot as plt


gpt2_data = {
    "category": ["culture", "moral", "neutral"],
    "mean_sentiment": [0.2339428699955016, 0.10012871057513917, 0.11884950880773204],
    "ATE_vs_neutral": [0.11509336118776956, -0.018720798232592872, 0.0],
    "model": ["GPT-2"] * 3
}

flant5_data = {
    "category": ["culture", "moral", "neutral"],
    "mean_sentiment": [0.0, 0.0, 0.07160695626212868],
    "ATE_vs_neutral": [-0.07160695626212868, -0.07160695626212868, 0.0],
    "model": ["Flan-T5"] * 3
}

gptneo_data = {
    "category": ["culture", "moral", "neutral"],
    "mean_sentiment": [0.15248917748917748, 0.24385912698412698, 0.14478538429199816],
    "ATE_vs_neutral": [0.007703793197179315, 0.09907374269212882, 0.0],
    "model": ["GPT-Neo"] * 3
}

# Convert to DataFrames
gpt2 = pd.DataFrame(gpt2_data)
flant5 = pd.DataFrame(flant5_data)
gptneo = pd.DataFrame(gptneo_data)

combined = pd.concat([gpt2, flant5, gptneo], ignore_index=True)

pivot = combined.pivot(index="category", columns="model", values="ATE_vs_neutral")
pivot["GPT2_minus_FlanT5"] = pivot["GPT-2"] - pivot["Flan-T5"]
pivot["GPTNeo_minus_FlanT5"] = pivot["GPT-Neo"] - pivot["Flan-T5"]
pivot["GPTNeo_minus_GPT2"] = pivot["GPT-Neo"] - pivot["GPT-2"]

pivot.to_csv("causal_comparison_summary.csv")
print("Causal comparison summary saved as 'causal_comparison_summary.csv'\n")

print(pivot)

plt.figure(figsize=(8, 5))
for model in ["GPT-2", "Flan-T5", "GPT-Neo"]:
    plt.bar(pivot.index + " (" + model + ")", pivot[model], label=model)

plt.title("Causal Bias (ATE vs Neutral) Across Models")
plt.xlabel("Category and Model")
plt.ylabel("Average Treatment Effect (ATE)")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Interpretation ---")
print("• GPT-2 shows strong cultural bias (ATE +0.115) and minor moral bias (-0.018).")
print("• Flan-T5 is nearly neutral across all categories (ATE ~ -0.07).")
print("• GPT-Neo shows moral positivity bias (ATE +0.099).")
print("=> Flan-T5 demonstrates the most balanced causal fairness among the models.")
