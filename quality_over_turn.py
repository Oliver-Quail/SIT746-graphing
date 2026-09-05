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

df_cleaned_reasoning_dq = reasoning.dropna(subset=['Decision Quality'])
df_cleaned_non_reasoning_dq = non_reasoning.dropna(subset=['Decision Quality'])


fig, ax = plt.subplots(figsize=(12,6), ncols=2, nrows=4)

labels = ["Turn 1", "Turn 2", "Turn 3", "Turn 4"]

# Row 1
ax[0,0].boxplot([df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)
ax[0,1].boxplot([df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)
# Row 2
ax[1,0].boxplot([df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)
ax[1,1].boxplot([df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_reasoning_dq.loc[(df_cleaned_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_reasoning_dq["Strategy"] == "Context") & (df_cleaned_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)

# Row 1
ax[2,0].boxplot([df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)
ax[2,1].boxplot([df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Zero") & (df_cleaned_non_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)
# Row 2
ax[3,0].boxplot([df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Simple") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)
ax[3,1].boxplot([df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 1)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 2)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 3)]["Decision Quality"], df_cleaned_non_reasoning_dq.loc[(df_cleaned_non_reasoning_dq["Scenario"] == "Complex") & (df_cleaned_non_reasoning_dq["Strategy"] == "Context") & (df_cleaned_non_reasoning_dq["Turn"] == 4)]["Decision Quality"]], tick_labels=labels)

ax[0,0].set_title("Q-0-S")
ax[0,1].set_title("Q-0-C")
ax[1,0].set_title("Q-C-S")
ax[1,1].set_title("Q-C-C")
ax[2,0].set_title("M-0-S")
ax[2,1].set_title("M-0-C")
ax[3,0].set_title("M-C-S")
ax[3,1].set_title("M-C-C")


ax[0,0].set_yticks([0,1,2,3,4,5])
ax[0,1].set_yticks([0,1,2,3,4,5])
ax[1,0].set_yticks([0,1,2,3,4,5])
ax[1,1].set_yticks([0,1,2,3,4,5])
ax[2,0].set_yticks([0,1,2,3,4,5])
ax[2,1].set_yticks([0,1,2,3,4,5])
ax[3,0].set_yticks([0,1,2,3,4,5])
ax[3,1].set_yticks([0,1,2,3,4,5])


plt.suptitle("Decision Quaility over time", x=0.5, ha='center') 
plt.tight_layout()
plt.savefig("graph/quality_over_time.png")