import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')

df = pd.read_csv("./Reesults.csv")



reasoning = df.loc[df["Model"] == "Qwen"]
non_reasoning = df.loc[df["Model"] == "Mistral"]





fig, ax = plt.subplots(figsize=(12,12), ncols=3, nrows=4)

df_cleaned_reasoning_faithfulness = reasoning.dropna(subset=['Faithfulness'])
df_cleaned_non_reasoning_faithfulness = non_reasoning.dropna(subset=['Faithfulness'])


df_cleaned_reasoning_coherence = reasoning.dropna(subset=['Coherence'])
df_cleaned_non_reasoning_coherence = non_reasoning.dropna(subset=['Coherence'])

df_cleaned_reasoning_dq = reasoning.dropna(subset=['Decision Quality'])
df_cleaned_non_reasoning_dq = non_reasoning.dropna(subset=['Decision Quality'])


# Row one
ax1 = ax[0,0].boxplot([df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Faithfulness"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[0,0].set_title("Faithfulness")

ax2 = ax[0,1].boxplot([df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Coherence"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Coherence"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[0,1].set_title("Coherence")

ax3 = ax[0,2].boxplot([df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Decision Quality"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[0,2].set_title("Decision Quality")

# Row two
ax4 = ax[1,0].boxplot([df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Faithfulness"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[1,0].set_title("Faithfulness")

ax5 = ax[1,1].boxplot([df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Coherence"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Coherence"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[1,1].set_title("Coherence")

ax6 = ax[1,2].boxplot([df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Decision Quality"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[1,2].set_title("Decision Quality")

# Row three
ax7 = ax[2,0].boxplot([df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Faithfulness"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[2,0].set_title("Faithfulness")

ax8 = ax[2,1].boxplot([df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Coherence"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Coherence"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[2,1].set_title("Coherence")

ax9 = ax[2,2].boxplot([df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Decision Quality"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[2,2].set_title("Decision Quality")

# Row four
ax10 = ax[3,0].boxplot([df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Faithfulness"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[3,0].set_title("Faithfulness")

ax11 = ax[3,1].boxplot([df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Coherence"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Coherence"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[3,1].set_title("Coherence")

ax12 = ax[3,2].boxplot([df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Decision Quality"]], tick_labels=["Qwen", "Mistral"], patch_artist=True)
ax[3,2].set_title("Decision Quality")

colours = ["purple", "orange"]

for plot in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12):
    for patch, colour in zip(plot["boxes"], colours):
        patch.set_facecolor(colour)

# Rows limits 1
ax[0,0].set_yticks([0,1,2,3,4,5])
ax[0,1].set_yticks([0,1,2,3,4,5])
ax[0,2].set_yticks([0,1,2,3,4,5])

# Rows limits 2
ax[1,0].set_yticks([0,1,2,3,4,5])
ax[1,1].set_yticks([0,1,2,3,4,5])
ax[1,2].set_yticks([0,1,2,3,4,5])

# Rows limits 3
ax[2,0].set_yticks([0,1,2,3,4,5])
ax[2,1].set_yticks([0,1,2,3,4,5])
ax[2,2].set_yticks([0,1,2,3,4,5])

# Rows limits 4
ax[3,0].set_yticks([0,1,2,3,4,5])
ax[3,1].set_yticks([0,1,2,3,4,5])
ax[3,2].set_yticks([0,1,2,3,4,5])



plt.suptitle("Faithfulness, Coherence and Decision Quality by Model Overall", x=0.5, ha='center') 
plt.tight_layout()
plt.savefig("graph/complexity.png")