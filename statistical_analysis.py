
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scipy.stats import ttest_rel

# ============================================
# Load Results
# ============================================

results = pd.read_csv("benchmark_results.csv")

print("Loaded results:", results.shape)

# ============================================
# Split Classification and Regression
# ============================================

classification = results[results["Task"] == "classification"]
regression = results[results["Task"] == "regression"]


# ============================================
# Friedman Test Function (Fixed Version)
# ============================================

def run_friedman_test(df, metric):

    pivot = df.pivot_table(
        index="Dataset",
        columns="Model",
        values="Main_Metric",
        aggfunc="mean"
    )

    # ---- FIX FOR NaN ISSUE ----
    pivot = pivot.dropna(axis=0)
    pivot = pivot.dropna(axis=1)

    models = pivot.columns
    datasets = pivot.index

    data = [pivot[m].values for m in models]

    stat, p_value = friedmanchisquare(*data)

    friedman_result = pd.DataFrame({
        "friedman_statistic": [stat],
        "p_value": [p_value],
        "n_models": [len(models)],
        "n_datasets": [len(datasets)]
    })

    return pivot, friedman_result


# ============================================
# Rank Table Function
# ============================================

def compute_ranks(pivot, higher_is_better=True):

    if higher_is_better:
        ranks = pivot.rank(axis=1, ascending=False)
    else:
        ranks = pivot.rank(axis=1, ascending=True)

    avg_ranks = ranks.mean()

    rank_table = pd.DataFrame({
        "model": avg_ranks.index,
        "average_rank": avg_ranks.values
    })

    rank_table = rank_table.sort_values("average_rank")

    return rank_table


# ============================================
# Pairwise p-values
# ============================================

def pairwise_pvalues(pivot):

    models = pivot.columns

    p_matrix = pd.DataFrame(
        np.ones((len(models), len(models))),
        index=models,
        columns=models
    )

    for m1 in models:
        for m2 in models:

            if m1 == m2:
                continue

            scores1 = pivot[m1]
            scores2 = pivot[m2]

            t_stat, p_val = ttest_rel(scores1, scores2)

            p_matrix.loc[m1, m2] = p_val

    return p_matrix


# ============================================
# Classification Analysis
# ============================================

clf_pivot, clf_friedman = run_friedman_test(
    classification,
    metric="accuracy"
)

clf_ranks = compute_ranks(
    clf_pivot,
    higher_is_better=True
)

clf_pvalues = pairwise_pvalues(clf_pivot)

clf_friedman.to_csv("friedman_classification_fixed.csv", index=False)
clf_ranks.to_csv("model_ranks_classification_fixed.csv", index=False)
clf_pvalues.to_csv("p_values_classification_fixed.csv")

print("Classification Friedman Test")
print(clf_friedman)
print("\nClassification Rank Table")
print(clf_ranks)


# ============================================
# Regression Analysis
# ============================================

reg_pivot, reg_friedman = run_friedman_test(
    regression,
    metric="rmse"
)

reg_ranks = compute_ranks(
    reg_pivot,
    higher_is_better=False
)

reg_pvalues = pairwise_pvalues(reg_pivot)

reg_friedman.to_csv("friedman_regression_fixed.csv", index=False)
reg_ranks.to_csv("model_ranks_regression_fixed.csv", index=False)
reg_pvalues.to_csv("p_values_regression_fixed.csv")

print("\nRegression Friedman Test")
print(reg_friedman)

print("\nRegression Rank Table")
print(reg_ranks)


# ============================================
# Generate LaTeX Rank Table
# ============================================

def generate_latex_rank_table(rank_table, filename):

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{lc}\n"
    latex += "\\hline\n"
    latex += "Model & Average Rank \\\\\n"
    latex += "\\hline\n"

    for _, row in rank_table.iterrows():
        latex += f"{row['model']} & {row['average_rank']:.3f} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Average Model Ranks Across Datasets}\n"
    latex += "\\end{table}\n"

    with open(filename, "w") as f:
        f.write(latex)


generate_latex_rank_table(clf_ranks, "rank_table_classification.tex")
generate_latex_rank_table(reg_ranks, "rank_table_regression.tex")

print("\nLaTeX tables generated.")