import matplotlib.pyplot as plt
import numpy as np

models = ['Qwen2.5-7B', 'Qwen2.5-7B+Prompts+SSC', 'OpenPangu-7B', 'OpenPangu-7B+Prompts+SSC']
f1 = [24.20, 26.49, 21.36, 34.29]
recall = [16.05, 51.25, 16.56, 25.13]
precision = [49.23, 17.86, 30.08, 53.94]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x-width, f1, width, label='F1', color='#4682B4')
plt.bar(x, recall, width, label='Recall', color='#DC5858')
plt.bar(x+width, precision, width, label='Precision', color='#5A9BD5')

plt.xticks(x, models, rotation=20)
plt.ylabel('Score (%)')
plt.title('Goal Interpretation: All Metrics Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('gi_bar.pdf')