import matplotlib.pyplot as plt, matplotlib as mpl
print("Using rc:", mpl.matplotlib_fname())
print("font.family:", mpl.rcParams.get("font.family"))
print("font.sans-serif[:5]:", mpl.rcParams.get("font.sans-serif")[:5])

plt.figure()
plt.title("平均時間（週）/ プリント作成")
plt.xlabel("時間（h）")
plt.ylabel("学校名")
plt.text(0.5, 0.5, "日本語OK？Meiryo / Yu Gothic / Noto Sans JP", ha="center", va="center")
plt.savefig("font_smoketest.png", dpi=160, bbox_inches="tight")
print("Wrote font_smoketest.png")
