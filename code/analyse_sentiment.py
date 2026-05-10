import os
import matplotlib.pyplot as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")             

THEMES_IA = {"hood": "Hood/Ghetto", "love": "Love", "money": "Money"}
THEMES_HUM = {"life": "Hood/Ghetto", "love": "Love", "money": "Money"}
MODELS_IA = ["gemma", "mixtral", "qwen"]

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    scores = [analyzer.polarity_scores(line)["compound"] for line in lines]
    return round(sum(scores) / len(scores), 4)

print("=== ANALYSE SENTIMENT (VADER) ===\n")
print(f"Corpus détecté : {CORPUS_DIR}\n")
print("── Corpus IA ──")

results_ia = {}

for model in MODELS_IA:
    results_ia[model] = {}
    folder = os.path.join(CORPUS_DIR, "IA", model)
    if not os.path.exists(folder):
        print(f"  [!] Dossier introuvable : {folder}")
        continue
    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue
        for key, label in THEMES_IA.items():
            if key in filename.lower():
                path = os.path.join(folder, filename)
                text = read_file(path)
                score = sentiment_score(text)
                results_ia[model][label] = score
                sentiment_label = "positif" if score > 0.05 else ("négatif" if score < -0.05 else "neutre")
                print(f"  {model:10s} | {label:15s} | score={score:+.4f} ({sentiment_label})")

print("\n── Corpus Humain ──")

results_hum = {}

for folder_name, label in THEMES_HUM.items():
    folder = os.path.join(CORPUS_DIR, "humain", folder_name)
    if not os.path.exists(folder):
        print(f"  [!] Dossier introuvable : {folder}")
        continue
    all_text = ""
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            all_text += "\n" + read_file(os.path.join(folder, filename))
    if all_text.strip():
        score = sentiment_score(all_text)
        results_hum[label] = score
        sentiment_label = "positif" if score > 0.05 else ("négatif" if score < -0.05 else "neutre")
        print(f"  humain     | {label:15s} | score={score:+.4f} ({sentiment_label})")

themes = list(THEMES_IA.values())
x = np.arange(len(themes))
width = 0.18

fig, ax = plt.subplots(figsize=(11, 6))

colors_ia = ["#4C72B0", "#DD8452", "#55A868"]
for i, model in enumerate(MODELS_IA):
    values = [results_ia[model].get(t, 0) for t in themes]
    ax.bar(x + i * width, values, width, label=model.capitalize(), color=colors_ia[i])

values_hum = [results_hum.get(t, 0) for t in themes]
ax.bar(x + 3 * width, values_hum, width, label="Humain", color="#C44E52", hatch="//")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axhspan(0.05, 1, alpha=0.04, color="green", label="_positif")
ax.axhspan(-1, -0.05, alpha=0.04, color="red", label="_négatif")

ax.set_xlabel("Thème", fontsize=12)
ax.set_ylabel("Score de sentiment (VADER compound)", fontsize=12)
ax.set_title("Sentiment moyen par thème et source\n(IA vs Humain)", fontsize=14, fontweight="bold")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(themes, fontsize=11)
ax.set_ylim(-1, 1)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
output_path = os.path.join(PROJECT_ROOT, "sentiment_results.png")
plt.savefig(output_path, dpi=150)
plt.show()
print(f"\n Graphique sauvegardé : {output_path}")
