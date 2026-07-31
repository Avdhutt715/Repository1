import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/model_results.csv")

print(df)

plt.figure(figsize=(10,5))

plt.bar(df["Model"], df["Accuracy"])

plt.xticks(rotation=20)

plt.ylabel("Accuracy")

plt.title("Machine Learning Model Comparison")

plt.tight_layout()

plt.savefig("plots/model_comparison.png")

plt.show()