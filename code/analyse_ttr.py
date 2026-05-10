import os
import re
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")           

THEMES_IA = {"hood": "Hood/Ghetto", "love": "Love", "money": "Money"}
THEMES_HUM = {"life": "Hood/Ghetto", "love": "Love", "money": "Money"}
MODELS_IA = ["gemma", "mixtral", "qwen"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def compute_ttr(text):
    tokens = clean_text(text).split()
    if len(tokens) == 0:
        return 0, 0, 0
    types = set(tokens)
    ttr = len(types) / len(tokens)
    return round(ttr, 4), len(tokens), len(types)

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

print("=== ANALYSE TTR ===\n")
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
                ttr, tokens, types = compute_ttr(text)
                results_ia[model][label] = ttr
                print(f"  {model:10s} | {label:15s} | TTR={ttr:.4f} ({tokens} tokens, {types} types)")

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
            all_text += " " + read_file(os.path.join(folder, filename))
    if all_text.strip():
        ttr, tokens, types = compute_ttr(all_text)
        results_hum[label] = ttr
        print(f"  humain     | {label:15s} | TTR={ttr:.4f} ({tokens} tokens, {types} types)")

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

ax.set_xlabel("Thème", fontsize=12)
ax.set_ylabel("TTR (Type-Token Ratio)", fontsize=12)
ax.set_title("Richesse lexicale par thème et source\n(IA vs Humain)", fontsize=14, fontweight="bold")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(themes, fontsize=11)
ax.set_ylim(0, max(
    max((v for m in results_ia.values() for v in m.values()), default=0),
    max(results_hum.values(), default=0)
) * 1.2)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
output_path = os.path.join(PROJECT_ROOT, "ttr_results.png")
plt.savefig(output_path, dpi=150)
plt.show()
print(f"\n Graphique sauvegardé : {output_path}")
