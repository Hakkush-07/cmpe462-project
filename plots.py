import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

df = pd.read_excel("nba.xlsx")

guards_better = ["Assists Contribution", "3 Pointers Attempted Contribution"]
centers_better = ["Rebounds Contribution", "Blocks Contribution"]

low, high = 0.05, 0.95
for s in guards_better + centers_better:
    sx = list(df[s].quantile([low, high]))
    df = df.loc[(df[s] >= sx[0]) & (df[s] <= sx[1])]

centers = df[(df["Position"] == "Center") | (df["Position"] == "Center-Forward")]
guards = df[(df["Position"] == "Guard") | (df["Position"] == "Guard-Forward")]
forwards = df[(df["Position"] == "Forward") | (df["Position"] == "Forward-Guard") | (df["Position"] == "Forward-Center")]

size = 10
print(len(centers))
x1 = centers[guards_better[0]]
for i in range(1, len(guards_better)):
    x1 = x1 * centers[guards_better[i]]
y1 = centers[centers_better[0]]
for i in range(1, len(centers_better)):
    y1 = y1 * centers[centers_better[i]]
plt.scatter(x1, y1, c="blue", s=size)

# print(len(forwards))
# x2 = forwards[guards_better[0]]
# for i in range(1, len(guards_better)):
#     x2 = x2 * forwards[guards_better[i]]
# y2 = forwards[centers_better[0]]
# for i in range(1, len(centers_better)):
#     y2 = y2 * forwards[centers_better[i]]
# plt.scatter(x2, y2, c="yellow", s=size)

print(len(guards))
x3 = guards[guards_better[0]]
for i in range(1, len(guards_better)):
    x3 = x3 * guards[guards_better[i]]
y3 = guards[centers_better[0]]
for i in range(1, len(centers_better)):
    y3 = y3 * guards[centers_better[i]]
plt.scatter(x3, y3, c="red", s=size)

plt.xlabel("guards better")
plt.ylabel("centers better")
ca = plt.gca()
# ca.add_patch(Rectangle((-0.001, 0.01), 0.006, 0.08, facecolor=(0.0, 0.0, 1.0, 0.3)))
# ca.add_patch(Rectangle((0.01, -0.001), 0.062, 0.02, facecolor=(1.0, 0.0, 0.0, 0.3)))
# ca.add_patch(Rectangle((0.007, 0.02), 0.02, 0.06, facecolor=(1.0, 1.0, 0.0, 0.3)))

plt.show()
