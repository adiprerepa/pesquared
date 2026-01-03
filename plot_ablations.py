import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directory
os.makedirs('plots', exist_ok=True)

# Read the CSV
df = pd.read_csv('optimization_results.csv')

# Shorten model names for readability
df['short_model'] = df['model'].apply(lambda m: m.split('-')[0])

# Helper to save figures into plots/
def save(fig_name):
    plt.tight_layout()
    path = os.path.join('plots', fig_name)
    plt.savefig(path)
    plt.close()

# Common plotting function
def plot_bar(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, ci='sd')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    save(filename)

def plot_box(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save(filename)

def plot_scatter(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue='short_model', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    save(filename)

# 1A. Model vs. Runtime Improvement
plot_bar('short_model', 'globe_runtime_improvement',
         'Average Globe Runtime Improvement by Model', 'Model', 'Globe Runtime Improvement (%)',
         'model_vs_runtime_improvement.png')

# 1B. Model vs. Instruction Improvement
plot_bar('short_model', 'globe_instructions_improvement',
         'Average Globe Instructions Improvement by Model', 'Model', 'Globe Instructions Improvement (%)',
         'model_vs_instructions_improvement.png')

# 2A. Include Clang Remarks vs. Runtime Improvement
plot_box('include_clang_remarks', 'globe_runtime_improvement',
         'Impact of Include Clang Remarks on Globe Runtime Improvement', 'Include Clang Remarks', 'Globe Runtime Improvement (%)',
         'clang_remarks_vs_runtime.png')

# 2B. Include Clang Remarks vs. Instructions Improvement
plot_box('include_clang_remarks', 'globe_instructions_improvement',
         'Impact of Include Clang Remarks on Globe Instructions Improvement', 'Include Clang Remarks', 'Globe Instructions Improvement (%)',
         'clang_remarks_vs_instructions.png')

# 3A. Obfuscation Tier vs. Runtime Improvement
plot_bar('obfuscation_tier', 'globe_runtime_improvement',
         'Obfuscation Tier vs. Globe Runtime Improvement', 'Obfuscation Tier', 'Globe Runtime Improvement (%)',
         'obfuscation_vs_runtime.png')

# 3B. Obfuscation Tier vs. Instructions Improvement
plot_bar('obfuscation_tier', 'globe_instructions_improvement',
         'Obfuscation Tier vs. Globe Instructions Improvement', 'Obfuscation Tier', 'Globe Instructions Improvement (%)',
         'obfuscation_vs_instructions.png')

# 4A. Callee Depth vs. Runtime Improvement
plot_scatter('callee_depth', 'globe_runtime_improvement',
             'Callee Depth vs. Globe Runtime Improvement', 'Callee Depth', 'Globe Runtime Improvement (%)',
             'callee_depth_vs_runtime.png')

# 4B. Callee Depth vs. Instructions Improvement
plot_scatter('callee_depth', 'globe_instructions_improvement',
             'Callee Depth vs. Globe Instructions Improvement', 'Callee Depth', 'Globe Instructions Improvement (%)',
             'callee_depth_vs_instructions.png')

# 5A. Function Length vs. Runtime Improvement
plot_scatter('function_length', 'globe_runtime_improvement',
             'Function Length vs. Globe Runtime Improvement', 'Function Length (Lines of Code)', 'Globe Runtime Improvement (%)',
             'function_length_vs_runtime_improvement.png')

# 5B. Function Length vs. Instructions Improvement
plot_scatter('function_length', 'globe_instructions_improvement',
             'Function Length vs. Globe Instructions Improvement', 'Function Length (Lines of Code)', 'Globe Instructions Improvement (%)',
             'function_length_vs_instructions_improvement.png')

# — Additional Interesting Plots —

# A. Pair-plot of improvement metrics
pairplot = sns.pairplot(df,
                        vars=[
                            'globe_cycles_improvement',
                            'globe_runtime_improvement',
                            'globe_instructions_improvement'
                        ],
                        hue='short_model',
                        corner=True,
                        plot_kws={'alpha': 0.5})
pairplot.savefig(os.path.join('plots', 'pairplot_improvements.png'))
plt.close('all')

# B. Correlation heatmap
numeric_cols = [
    'function_length',
    'function_relative_level',
    'obfuscation_tier',
    'callee_depth',
    'globe_cycles_improvement',
    'globe_runtime_improvement',
    'globe_instructions_improvement'
]
corr = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr,
            annot=True,
            fmt='.2f',
            cmap='vlag',
            cbar_kws={'shrink': .8})
plt.title('Correlation Matrix of Features & Improvements')
save('correlation_heatmap.png')

# C. Instruction Count Related

# 6. Globe Instruction Count vs. Runtime Improvement
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df,
                x='globe_instructions',
                y='globe_runtime_improvement',
                hue='short_model',
                alpha=0.7)
plt.title('Globe Instruction Count vs. Runtime Improvement')
plt.xlabel('Globe Instruction Count (Post-Optimization)')
plt.ylabel('Globe Runtime Improvement (%)')
plt.xscale('log')
save('instructions_vs_runtime_improvement.png')

# 7. Globe Instructions Improvement vs. Cycles Improvement
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df,
                x='globe_instructions_improvement',
                y='globe_cycles_improvement',
                hue='short_model',
                alpha=0.7)
plt.title('Globe Instructions Improvement vs. Cycles Improvement')
plt.xlabel('Globe Instructions Improvement (%)')
plt.ylabel('Globe Cycles Improvement (%)')
save('instruction_improvement_vs_cycles_improvement.png')

print("✅ All plots saved into the `plots/` directory, with both runtime AND instruction improvements!")
