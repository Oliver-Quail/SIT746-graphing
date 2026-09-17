import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')

df = pd.read_csv("./Reesults.csv")



reasoning = df.loc[df["Model"] == "Qwen"]
non_reasoning = df.loc[df["Model"] == "Mistral"]





fig, ax = plt.subplots(figsize=(12,18), ncols=1, nrows=3)

df_cleaned_reasoning_faithfulness = reasoning.dropna(subset=['Faithfulness'])
df_cleaned_non_reasoning_faithfulness = non_reasoning.dropna(subset=['Faithfulness'])


df_cleaned_reasoning_coherence = reasoning.dropna(subset=['Coherence'])
df_cleaned_non_reasoning_coherence = non_reasoning.dropna(subset=['Coherence'])

df_cleaned_reasoning_dq = reasoning.dropna(subset=['Decision Quality'])
df_cleaned_non_reasoning_dq = non_reasoning.dropna(subset=['Decision Quality'])

labels = ["Q-0-S", "M-0-S", "Q-0-C", "M-0-C", "Q-C-S", "M-C-S", "Q-C-C", "M-C-C"]

# Row one
ax1 = ax[0].boxplot([df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Faithfulness"], df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Faithfulness"], df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Faithfulness"], df_cleaned_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Faithfulness"], df_cleaned_non_reasoning_faithfulness.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Faithfulness"] ], tick_labels=labels, patch_artist=True)
ax[0].set_title("Faithfulness", fontsize=16)

ax2 = ax[1].boxplot([df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Faithfulness"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Coherence"], df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Coherence"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Coherence"], df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Coherence"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Coherence"], df_cleaned_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Coherence"], df_cleaned_non_reasoning_coherence.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Coherence"] ], tick_labels=labels, patch_artist=True)
ax[1].set_title("Coherence", fontsize=16)

ax3 = ax[2].boxplot([df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Zero")]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Zero")]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Simple") & (df["Strategy"] == "Context")]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df["Scenario"] == "Complex") & (df["Strategy"] == "Context")]["Decision Quality"] ], tick_labels=labels, patch_artist=True)
ax[2].set_title("Decision Quality", fontsize=16)


colours = ["purple", "orange", "purple", "orange", "purple", "orange", "purple", "orange"]

for plot in (ax1, ax2, ax3):
    for patch, colour in zip(plot["boxes"], colours):
        patch.set_facecolor(colour)

# Rows limits 1
ax[0].set_yticks([0,1,2,3,4,5])
ax[1].set_yticks([0,1,2,3,4,5])
ax[2].set_yticks([0,1,2,3,4,5])

# xticks rotation
ax[0].tick_params(axis='x', labelrotation=90)
ax[1].tick_params(axis='x', labelrotation=90)
ax[2].tick_params(axis='x', labelrotation=90)

ax[0].tick_params(axis="x", labelsize=16)
ax[0].tick_params(axis="y", labelsize=16)
ax[1].tick_params(axis="x", labelsize=16)
ax[1].tick_params(axis="y", labelsize=16)
ax[2].tick_params(axis="x", labelsize=16)
ax[2].tick_params(axis="y", labelsize=16)


ax[0].text(
    1, -0.2,                  
    "(a)",        
    transform=ax[0].transAxes,      
    verticalalignment='bottom',   
    horizontalalignment='right',  
    fontsize=16,
    color='Black'
)

ax[1].text(
    1, -0.2,                  
    "(b)",        
    transform=ax[1].transAxes,      
    verticalalignment='bottom',   
    horizontalalignment='right',  
    fontsize=16,
    color='Black'
)

ax[2].text(
    1, -0.2,                  
    "(c)",        
    transform=ax[2].transAxes,      
    verticalalignment='bottom',   
    horizontalalignment='right',  
    fontsize=16,
    color='Black'
)


plt.suptitle("Faithfulness, Coherence and Decision Quality by Model Overall") 
plt.tight_layout()
plt.savefig("graph/complexity_combined.png")