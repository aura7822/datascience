import matplotlib.pyplot as plt
import numpy as np


months = ["07/2019", "08/2019", "09/2019", "10/2019", "11/2019"]
searches = [50, 59, 42, 56, 62]
direct = [39, 47, 42, 51, 51]
social = [70, 80, 90, 87, 9]
x = np.arange(len(months))
width = 0.25  # width of each bar


fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x - width, searches, width, label="Searches", color="royalblue")
ax.bar(x, direct, width, label="Direct", color="violet")
ax.bar(x + width, social, width, label="Social Media", color="orange")

ax.set_title("Visitors by web traffic sources", fontsize=14, fontweight="bold")
ax.set_xlabel("months")
ax.set_ylabel("visitors in thousands")
ax.set_xticks(x)
ax.set_xticklabels(months)


ax.legend()

ax.yaxis.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
