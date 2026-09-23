import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#####-------------Finnish non financial listed companies for SimDec -------------
# Read Finnish non-financial listed companies dataset
Finnish_list_companies = pd.read_excel("thesis_Nylund.xlsx")

# Select continuous numerical variables
scatterplot_variables = [
    "market_cap",
#    "total_assets",
#    "total_debt",
    "DebtRatio",
#    "equity",
#    "net_income",
    "ROA",
    "book_to_market",
    "tobins_q"
]

# Create the scatterplot matrix
g = sns.pairplot(
    Finnish_list_companies,
    vars=scatterplot_variables,
    diag_kind="hist",
    plot_kws={
        "alpha": 0.5,
        "s": 20
    },
    diag_kws={
        "bins": 15,
        "color": "blue",
        "edgecolor": "black"
    }
)

# Adjust axis fonts
for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.tick_params(axis="both", labelsize=11)

# plt.suptitle(
#    "Scatterplot Matrix of Numerical Variables in the Finnish non Financial Listed Companies Dataset",
#    fontsize=18,
#    y=1.02
#)

plt.show()

# Select numerical input variables
numerical_inputs_finnish = [
    "market_cap",
    "total_assets",
    "total_debt",
    "DebtRatio",
    "equity",
    "net_income",
    "ROA",
    "book_to_market",
    "Derivatives nominal value",
    "Hedge ratio",
    "Policy Rate(average)",
    "Policy rate change (units)"
]

# Calculate correlations between numerical inputs
input_correlation_finnish = (
    Finnish_list_companies[numerical_inputs_finnish].corr()
)

# Plot the correlation heatmap
plt.figure(figsize=(14, 11))

sns.heatmap(
    input_correlation_finnish,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 13},
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

# plt.title("(b) Correlation matrix of numerical inputs in the Finnish non-financial listed companies dataset",fontsize=20,pad=15)
plt.xticks(rotation=45, ha="right", fontsize=15)
plt.yticks(rotation=0, fontsize=15)
plt.tight_layout()
plt.show()

# Tobin's Q distribution
plt.figure(figsize=(9, 6))
plt.hist(Finnish_list_companies["tobins_q"], bins=41,weights=(np.ones(len(Finnish_list_companies)) / len(Finnish_list_companies)), edgecolor="black")
plt.xlabel("Tobin's Q")
plt.ylabel("Probability")
plt.title("(b) Distribution of tobin's Q in the Finnish non-financial listed companies dataset", fontsize=14, pad=15)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Convert categorical variables into numerical codes
categorical_columns = Finnish_list_companies.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:
    Finnish_list_companies[column] = (
        Finnish_list_companies[column]
        .astype("string")
        .str.strip()
        .astype("category")
        .cat.codes
    )

# Clean all column names
Finnish_list_companies.columns = (
    Finnish_list_companies.columns
    .astype(str)
    .str.strip()
    .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_")
)

numeric_data = Finnish_list_companies.select_dtypes(include="number")

# Save with a completely new and simple filename
output_file = "simdec_finnish_list_companies.csv"

Finnish_list_companies.to_csv(
    output_file,
    index=False,
    sep=",",
    encoding="ascii",
    lineterminator="\n"
)

