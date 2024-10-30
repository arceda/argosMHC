import pandas as pd
import matplotlib.pyplot as plt

figType = "loss"
model="t6"
c = ["c1", "c2", "c3"]

df = pd.read_csv(f'{figType}_classic-{model}_c1-c2-c3.csv')

x_values = df['train/global_step']
y_values_c3 = df[f'esm_{model}_{c[0]} - eval/'+figType.lower()]
y_values_c4 = df[f'esm_{model}_{c[1]} - eval/'+figType.lower()]
y_values_c5 = df[f'esm_{model}_{c[2]} - eval/'+figType.lower()]

plt.figure(figsize=(8, 5), dpi=300)

plt.plot(x_values, y_values_c3, label=f'ESM2({model})-{c[0]}', marker='o')
plt.plot(x_values, y_values_c4, label=f'ESM2({model})-{c[1]}', marker='o')
plt.plot(x_values, y_values_c5, label=f'ESM2({model})-{c[2]}', marker='o')

plt.xlabel('train/global_step')
plt.ylabel('eval/loss')
#plt.title(figType+f' Comparison: classic-{model}-{c[0]}, classic-{model}-{c[1]}, classic-{model}-{c[2]}')

plt.legend()

plt.savefig(f'{figType}_comparison-{model}.eps', format='eps')

plt.show()