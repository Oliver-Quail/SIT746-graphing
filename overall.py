import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')

df = pd.read_csv("./Reesults.csv")


print(df.info())


reasoning = df.loc[df["Model"] == "Qwen"]
non_reasoning = df.loc[df["Model"] == "Mistral"]





fig, ax = plt.subplots(figsize=(12,6), ncols=3, nrows=1)

df_cleaned_reasoning_faithfulness = reasoning.dropna(subset=['Faithfulness'])
df_cleaned_non_reasoning_faithfulness = non_reasoning.dropna(subset=['Faithfulness'])

df_cleaned_reasoning_coherence = reasoning.dropna(subset=['Coherence'])
df_cleaned_non_reasoning_coherence = non_reasoning.dropna(subset=['Coherence'])

df_cleaned_reasoning_dq = reasoning.dropna(subset=['Decision Quality'])
df_cleaned_non_reasoning_dq = non_reasoning.dropna(subset=['Decision Quality'])


ax1 = ax[0].boxplot([df_cleaned_reasoning_faithfulness["Faithfulness"], df_cleaned_non_reasoning_faithfulness["Faithfulness"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[0].set_title("Faithfulness")

ax2 = ax[1].boxplot([df_cleaned_reasoning_coherence["Coherence"], df_cleaned_non_reasoning_coherence["Coherence"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[1].set_title("Coherence")

ax3 = ax[2].boxplot([df_cleaned_reasoning_dq["Decision Quality"], df_cleaned_non_reasoning_dq["Decision Quality"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[2].set_title("Decision Quality")

colours = ["purple", "orange"]

for plot in (ax1, ax2, ax3):
    for patch, colour in zip(plot["boxes"], colours):
        patch.set_facecolor(colour)

plt.suptitle("Faithfulness, Coherence and Decision Quality by Model Overall", x=0.5, ha='center') 
plt.tight_layout()
plt.savefig("overall.png")