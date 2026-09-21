import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
    )
import seaborn as sns

# Read Bank marketing dataset
bank_marketing = pd.read_csv("bank-full.csv", sep = ";")

# Check the basic information about the dataset
print(bank_marketing.head())
print(bank_marketing.shape)
print(bank_marketing.columns)
print(bank_marketing.isnull().sum())
print(bank_marketing.columns)

# Select numerical variables for the scatterplot matrix
scatterplot_variables = [
    "age",
    "balance",
    "duration",
    "campaign",
    "previous"
]

# Create the scatterplot matrix
g = sns.pairplot(
    bank_marketing,
    vars=scatterplot_variables,
    hue="y",
    diag_kind="hist",
    palette={
        "no": "blue",
        "yes": "red"
    },
    plot_kws={
        "alpha": 0.4,
        "s": 15
    },
    diag_kws={
        "bins": 15,
        "alpha": 0.5,
        "edgecolor": "black"
    }
)

# Adjust axis fonts
for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlabel(
            ax.get_xlabel(),
            fontsize=15
        )
        ax.set_ylabel(
            ax.get_ylabel(),
            fontsize=15
        )
        ax.tick_params(
            axis="both",
            labelsize=10
        )

# Place the legend at the upper-right of the entire figure
if g._legend is not None:
    g._legend.set_title(
        "Subscription",
        prop={"size": 15}
    )

    for text in g._legend.texts:
        text.set_fontsize(15)

    g._legend.set_loc("upper left")
    g._legend.set_bbox_to_anchor((0.85, 1.00))

# Leave space on the right for the legend
g.figure.subplots_adjust(
    right=0.84,
    top=0.93
)

plt.show()

# Data preparation
# Separate X and Y
X = bank_marketing.drop(columns = "y")
Y = bank_marketing["y"]

# Identify numerical and categorical variables
numerical_variables = X.select_dtypes(include = "number").columns.tolist()
categorical_variables = X.select_dtypes(exclude = "number").columns.tolist()

print("Numerical variables:")
print(numerical_variables)
print("\nCategorical variables:")
print(categorical_variables)

# Summary statistics for numerical variables
print(X[numerical_variables].describe().T)
# Summary for categorical variables
print(X[categorical_variables].describe().T)

# Transform the target variable (Y) into numbers
Y_numeric = Y.map({"no": 0, "yes": 1})

# Create a copy of X for numerical encoding
X_numeric = X.copy()
# Store the coding rule for each categorical variable
category_mappings ={}

# Transform all categorical X variables into numbers
for variable in categorical_variables:
    categories = sorted(X_numeric[variable].unique())
    mapping = {
        category: code
        for code, category in enumerate(categories, start=1)
    }

    category_mappings[variable] = mapping
    X_numeric[variable] = X_numeric[variable].map(mapping)

# Display the coding rules
for variable, mapping in category_mappings.items():
    print(f"\n{variable}:")
    print(mapping)
# Check the transformed X variables
print("\nTransformed X variables:")
print(X_numeric.head())
print(X_numeric.dtypes)

# Check whether all X variables are numerical
print(
    "\nAll X variables are numerical:",
    all(X_numeric.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype)))
    )
# Combine the transformed X and Y variables
bank_marketing_numeric = X_numeric.copy()
bank_marketing_numeric["y"] = Y_numeric

# Split the data into traning and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y_numeric,
    test_size = 0.4,
    random_state = 0 # make the result repeatable
    )

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining set class distribution:")
print(Y_train.value_counts())
print(Y_train.value_counts(normalize=True).mul(100))

print("\nTest set class distribution:")
print(Y_test.value_counts())
print(Y_test.value_counts(normalize=True).mul(100))

# Define preprocessing for numerical and categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_variables),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_variables)
    ]
    )

# Learn the preprocessing rules from the training set
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same rules to the test set
X_test_processed = preprocessor.transform(X_test)

# Display the shapes after preprocessing
print("Processed training set shape:", X_train_processed.shape)
print("Processed test set shape:", X_test_processed.shape)

# Check the processed values
print("\nFirst five processed training observations:")
print(X_train_processed[:5])

# Build the SVM model with an RBF kernel
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")

# Train the SVM model
svm_model.fit(X_train_processed, Y_train)

# Predict the test data
Y_svm = svm_model.predict(X_test_processed)

# Evaluation SVM performance
# Obtain decision scores for AUC
Y_svm_score = svm_model.decision_function(X_test_processed)

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, Y_svm, labels = [1,0])

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, Y_svm)
precision = precision_score(Y_test, Y_svm)
recall = recall_score(Y_test, Y_svm)
f1 = f1_score(Y_test, Y_svm)
auc = roc_auc_score(Y_test, Y_svm_score)

print("Confusion matrix:")
print(cm)

print("\nSVM evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# Convert the confusion matrix into row percentages
cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Confusion matrix with counts
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Subscription", "Non-subscription"]
    ).plot(
    ax=axes[0],
    cmap="Blues",
    colorbar=False,
    values_format="d"
    )

axes[0].set_title("Counts")

# Row-normalised confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm_percentage,
    display_labels=["Subscription", "Non-subscription"]
    ).plot(
    ax=axes[1],
    cmap="Blues",
    colorbar=False,
    values_format=".1f"
    )

fig.suptitle("Confusion Matrix for SVM")
axes[1].set_title("Row Percentages (%)")
# Rotate and centre the y-axis tick labels
for ax in axes:
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        ha="center",
        va="center",
        rotation_mode="anchor"
    )
    ax.tick_params(axis="y", pad=12)

plt.tight_layout()
plt.show()

# Plot ROC curve
roc_display = RocCurveDisplay.from_predictions( Y_test, Y_svm_score)

roc_display.ax_.set_xlabel("False Positive Rate")
roc_display.ax_.set_ylabel("True Positive Rate")
roc_display.ax_.set_title("ROC Curve for the SVM Model")
roc_display.ax_.grid(alpha=0.3)

plt.show()

# Create a table of SVM performance measures
performance_table = pd.DataFrame({
    "Performance measure": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
    })

performance_table["Value"] = performance_table["Value"].round(4)

print("\nSVM performance measures:")
print(performance_table.to_string(index=False))

###----------------------Bank Marketing for SimDec ---------------------------
# Read the Bank Marketing dataset
bank_marketing = pd.read_csv("bank-full.csv", sep=";")

# Numerical input vairbales
numerical_inputs_bank =[
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
    ]

# Calculate correlations between numerical inputs
input_correlation_bank =(
    bank_marketing[numerical_inputs_bank].corr()
    )

# Plot correlation heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(input_correlation_bank, annot = True, fmt=".2f", 
            annot_kws={"size": 12}, 
            cmap="coolwarm", vmin=-1, vmax=1, square=True)

#plt.title("(c) Correlation matrix of numerical inputs in the bank marketing dataset", fontsize=16,pad=15)

plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.show()

# Identify categorical input variables
categorical_variables_bank = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

# Print the categories of each categorical variable
for variable in categorical_variables_bank:
    categories = sorted(
        bank_marketing[variable].dropna().unique()
    )
    
    print(variable)
    print(categories)
    print("Number of categories:", len(categories))
    print()

# Identify categorical variables
categorical_variables = (
    bank_marketing
    .select_dtypes(include=["object", "string", "category"])
    .columns
    .tolist()
)

# Remove extra spaces
for variable in categorical_variables:
    bank_marketing[variable] = (
        bank_marketing[variable]
        .astype("string")
        .str.strip()
    )

# Convert the output into binary form
bank_marketing["y"] = (
    bank_marketing["y"]
    .map({"no": 0, "yes": 1})
    .astype("Int64")
)

# Calculate the mean subscription rate for each poutcome category
poutcome_means = (
    bank_marketing
    .groupby("poutcome")["y"]
    .mean()
    .sort_values()
)

print("Mean subscription rate by poutcome:")
print(poutcome_means)

# Assign codes in ascending order of the mean subscription rate
poutcome_mapping = {
    category: code
    for code, category in enumerate(poutcome_means.index)
}

print("\npoutcome coding:")
print(poutcome_mapping)

bank_marketing["poutcome"] = (
    bank_marketing["poutcome"]
    .map(poutcome_mapping)
    .astype("Int64")
)

# Store the coding rules
category_mappings = {
    "poutcome": poutcome_mapping,
    "y": {"no": 0, "yes": 1}
}

# Select the remaining categorical variables
remaining_categorical_variables = [
    variable
    for variable in categorical_variables
    if variable not in ["poutcome", "y"]
]

# Convert the remaining categorical variables alphabetically
for variable in remaining_categorical_variables:
    categories = sorted(
        bank_marketing[variable].dropna().unique()
    )

    mapping = {
        category: code
        for code, category in enumerate(categories)
    }

    category_mappings[variable] = mapping

    bank_marketing[variable] = (
        bank_marketing[variable]
        .map(mapping)
        .astype("Int64")
    )
    
print(poutcome_means)
print(poutcome_mapping)
    
# Target binary y distribution
plt.figure(figsize=(9, 6))
plt.hist(bank_marketing["y"], bins=11,  weights=np.ones(len(bank_marketing)) / len(bank_marketing), edgecolor="black")
# Replace numerical codes with category names
plt.xticks(
    [0, 1],
    ["No", "Yes"]
)
plt.xlabel("The target binary y")
plt.ylabel("Probability")
plt.title("(c) Distribution of the target binary y in the bank marketing dataset", fontsize=14, pad=15)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()    
    

# Save the numerical dataset for SimDec
bank_marketing.to_csv("simdec_bank_marketing.csv", index=False)

