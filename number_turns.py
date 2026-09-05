import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')

df = pd.read_csv("./Reesults.csv")



reasoning = df.loc[df["Model"] == "Qwen"]
non_reasoning = df.loc[df["Model"] == "Mistral"]



turn_counts = df.groupby(["Model", "Scenario", "Strategy", "Test"])["Turn"].size().reset_index(name='Total Turns')


turn_counts = turn_counts.dropna(subset=['Total Turns'])

turn_counts["Total Turns"] = turn_counts["Total Turns"]/3
print(turn_counts)

turn_count_non_reasoning = turn_counts.loc[turn_counts["Model"]  == "Mistral"]
turn_count_reasoning = turn_counts.loc[turn_counts["Model"]  == "Qwen"]

print(turn_count_reasoning)

fig, ax = plt.subplots(figsize=(18,6))


labels = ["Q-0-S", "M-0-S", "Q-0-C", "M-0-C", "Q-C-S", "M-C-S", "Q-C-C", "M-C-C"]


# Row one
ax1 = ax.boxplot([turn_count_reasoning.loc[(turn_count_reasoning["Scenario"] == "Simple") & (turn_count_reasoning["Strategy"] == "Zero")]["Total Turns"], turn_count_non_reasoning.loc[(turn_count_non_reasoning["Scenario"] == "Simple") & (turn_count_non_reasoning["Strategy"] == "Zero")]["Total Turns"], turn_count_reasoning.loc[(turn_count_reasoning["Scenario"] == "Complex") & (turn_count_reasoning["Strategy"] == "Zero")]["Total Turns"], turn_count_non_reasoning.loc[(turn_count_non_reasoning["Scenario"] == "Complex") & (turn_count_non_reasoning["Strategy"] == "Zero")]["Total Turns"], turn_count_reasoning.loc[(turn_count_reasoning["Scenario"] == "Simple") & (turn_count_reasoning["Strategy"] == "Context")]["Total Turns"], turn_count_non_reasoning.loc[(turn_count_non_reasoning["Scenario"] == "Simple") & (turn_count_non_reasoning["Strategy"] == "Context")]["Total Turns"], turn_count_reasoning.loc[(turn_count_reasoning["Scenario"] == "Complex") & (turn_count_reasoning["Strategy"] == "Context")]["Total Turns"], turn_count_non_reasoning.loc[(turn_count_non_reasoning["Scenario"] == "Complex") & (turn_count_non_reasoning["Strategy"] == "Context")]["Total Turns"] ], tick_labels=labels, patch_artist=True)
ax.set_title("Faithfulness")



colours = ["purple", "orange", "purple", "orange", "purple", "orange", "purple", "orange"]


for patch, colour in zip(ax1["boxes"], colours):
    patch.set_facecolor(colour)

# Rows limits 1
ax.set_yticks([0,1,2,3,4])

# xticks rotation
ax.tick_params(axis='x', labelrotation=90)



plt.suptitle("Total Turns, Coherence and Decision Quality by Model Overall", x=0.5, ha='center') 
plt.tight_layout()
plt.savefig("graph/number_of_turns.png")