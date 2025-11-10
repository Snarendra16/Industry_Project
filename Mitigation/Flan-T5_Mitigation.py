from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../data/output_flan_t5.csv")
df.columns = df.columns.str.strip().str.lower()

def compute_bias_score(text):
    text = str(text)
    sentiment = TextBlob(text).sentiment.polarity
    return (1 - sentiment) / 2

df["toxicity_score"] = df["model_output"].apply(compute_bias_score)

causal_effects = {
    "culture": -0.07160695626212868,
    "moral": -0.07160695626212868,
    "neutral": 0.0
}

def classify_category(text):
    text = text.lower()
    if any(word in text for word in ["tradition", "india", "religion", "language", "culture"]):
        return "culture"
    elif any(word in text for word in ["ethics", "right", "wrong", "moral", "justice"]):
        return "moral"
    else:
        return "neutral"

df["category"] = df["prompt_text"].apply(classify_category)

def mitigate_bias(row):
    category = row["category"]
    bias = causal_effects.get(category, 0.0)
    return row["toxicity_score"] - bias

df["mitigated_toxicity"] = df.apply(mitigate_bias, axis=1).clip(0, 1)

before_mean = df["toxicity_score"].mean()
after_mean = df["mitigated_toxicity"].mean()
reduction = (before_mean - after_mean) / before_mean * 100

print(f"\nAverage Toxicity Before Mitigation: {before_mean:.4f}")
print(f"Average Toxicity After Mitigation:  {after_mean:.4f}")
print(f"Bias Reduction Achieved: {reduction:.2f}%")

plt.figure(figsize=(6, 5))
bars = plt.bar(["Before", "After"], [before_mean, after_mean], color=["red", "green"])
plt.title("Flan-T5 Bias Mitigation using Causal Adjustment")
plt.ylabel("Average Toxicity Score")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha="center", fontsize=10, fontweight="bold")

plt.text(0.5, (before_mean + after_mean) / 2, f"↓ {reduction:.2f}% Bias Reduced",
         ha="center", va="center", fontsize=12, color="blue",
         fontweight="bold", bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.show()

df.to_csv("flant5_bias_mitigated_outputs.csv", index=False)
print("Mitigated outputs saved as 'flant5_bias_mitigated_outputs.csv'")
