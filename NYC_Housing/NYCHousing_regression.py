import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.stattools import jarque_bera
import seaborn as sns

# Read the NYC Calendar Sales 2015 dataset
NYC_housing = pd.read_csv("NYCHousing2015.csv")

# Check the basic information about the dataset
print(NYC_housing.head())
print(NYC_housing.shape)
print(NYC_housing.dtypes)
print(NYC_housing.isnull().sum())
print(NYC_housing.columns)

# Convert BUILDING CLASS CATEGORY into 1,2,3 codes
NYC_housing["BUILDING CLASS CATEGORY"] = pd.to_numeric(NYC_housing["BUILDING CLASS CATEGORY"].astype("string").str.strip().str.extract(r"^(\d{2})", expand=False), errors="coerce")
NYC_housing = NYC_housing[NYC_housing["BUILDING CLASS CATEGORY"].isin([1, 2, 3])].copy()

# Convert SALE DATE to datetime and retain only month and day
NYC_housing["SALE DATE"] = pd.to_datetime(NYC_housing["SALE DATE"],errors="coerce")
NYC_housing["MM"] = NYC_housing["SALE DATE"].dt.month
NYC_housing["DD"] = NYC_housing["SALE DATE"].dt.day
NYC_housing = NYC_housing.drop(columns="SALE DATE")

# Check the outliers in SALE PRICE 
# Histogram of SALE PRICE
plt.figure(figsize = (8,5))
plt.hist(NYC_housing["SALE PRICE"], bins=41, edgecolor="black")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.title("(a) Distribution of sale price before outlier removal")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Boxplot of SALE PRICE
plt.figure(figsize=(8,2.5))
plt.boxplot(NYC_housing["SALE PRICE"], vert=False)
plt.xlabel("Sale Price")
plt.title("Boxplot of Sale Price")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Descriptive statistics for all numeric variables
descriptive_summary = NYC_housing.describe().round(2)
print(descriptive_summary)

# Median and MAD to remove outliters in SALE PRICE
# Rule: median ± 3 × scaled MAD
sale_price = NYC_housing["SALE PRICE"]
median_price = sale_price.median()
mad = np.median(np.abs(sale_price.dropna() - median_price))

# MATLAB scales MAD to make it consistent with standard deviation
scaled_mad = mad / 0.6744897501960817
lower_limit = median_price - 3 * scaled_mad
upper_limit = median_price + 3 * scaled_mad

# Identify outliers
price_outliers = (
    (sale_price < lower_limit) |
    (sale_price > upper_limit)
    )

print("Median:", median_price)
print("MAD:", mad)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Number of outliers:", price_outliers.sum())

# Remove SALE PRICES outliers
NYC_housing = NYC_housing[~price_outliers].copy()

# Histogram of SALE PRICE
plt.figure(figsize = (8,5))
plt.hist(NYC_housing["SALE PRICE"], bins=41, edgecolor="black")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.title("(b) Distribution of sale price after outlier removal")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Check missing values
missing_values = pd.DataFrame({
    "Missing": NYC_housing.isna().sum(),
    "Percentage": NYC_housing.isna().mean() * 100
    })
print(missing_values.round(2))

# Convert numerical variables to numeric format
numeric_variables = [
#    "RESIDENTIAL UNITS",
#    "COMMERCIAL UNITS",
    "LAND SQUARE FEET",
    "GROSS SQUARE FEET",
#    "YEAR BUILT",
    "SALE PRICE"
    ]

for variable in numeric_variables:
    NYC_housing[variable] = pd.to_numeric(
        NYC_housing[variable],
        errors="coerce"
    )

g = sns.pairplot(
    NYC_housing,
    vars=numeric_variables,
    diag_kind="hist",
    plot_kws={
        "alpha": 0.3,
        "s": 12
    },
    diag_kws={
        "bins": 30,
        "color": "blue",
        "edgecolor": "black"
    }
)

# Increase axis-label and tick-label sizes
for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=11)
        ax.set_ylabel(ax.get_ylabel(), fontsize=11)
        ax.tick_params(axis="both", labelsize=9)

#plt.suptitle(
#    "Scatterplot Matrix of Numerical Variables in the New York City Housing Sales Dataset",
#    fontsize=18,
#    y=1.02
#)

plt.show()

# Check the correlations between continuous variables
corr_vars = [
    "RESIDENTIAL UNITS",
    "COMMERCIAL UNITS",
    "LAND SQUARE FEET",
    "GROSS SQUARE FEET",
    "YEAR BUILT",
    "MM",
    "DD"
    ]

# Correlation matrix
corr_matrix = NYC_housing[corr_vars].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr_vars)), corr_vars, rotation=45, ha="right")
plt.yticks(range(len(corr_vars)), corr_vars)

# Show correlation coefficients
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        plt.text(j, i,
                 f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center",
                 fontsize=8)
plt.title("Pearson correlation matrix")
plt.tight_layout()
plt.show()

# Convert categorical variables to dummy variables
NYC_housing = pd.get_dummies(
    NYC_housing,
    columns=["BOROUGH", "BUILDING CLASS CATEGORY"],
    drop_first=True,
    dtype=int
    )

# Check the new variables
print(NYC_housing.columns)

print(NYC_housing.head())
print(NYC_housing.dtypes)

# Fit the full multiple linear regression model
full_model = smf.ols(
    formula='''
Q("SALE PRICE") ~
Q("RESIDENTIAL UNITS") +
Q("COMMERCIAL UNITS") +
Q("LAND SQUARE FEET") +
Q("GROSS SQUARE FEET") +
Q("YEAR BUILT") +
MM +
DD +
Q("BOROUGH_Brooklyn") +
Q("BOROUGH_Manhattan") +
Q("BOROUGH_Queens") +
Q("BOROUGH_Staten Island") +
Q("BUILDING CLASS CATEGORY_2") +
Q("BUILDING CLASS CATEGORY_3")
''',
    data=NYC_housing
).fit()

# Display regression results
print(full_model.summary())

# Regression coefficients
full_model_coefficients = pd.DataFrame({
    "Coefficient": full_model.params,
    "Standard Error": full_model.bse,
    "t-value": full_model.tvalues,
    "p-value": full_model.pvalues,
    "95% CI Lower": full_model.conf_int()[0],
    "95% CI Upper": full_model.conf_int()[1]
})
print(full_model_coefficients.round(4))

# Full model summary
full_model_summary = pd.DataFrame({
    "Statistic": [
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "Prob (F-statistic)",
        "AIC",
        "BIC",
        "Number of observations",
        "Degrees of freedom (Model)",
        "Degrees of freedom (Residual)"
    ],
    "Value": [
        full_model.rsquared,
        full_model.rsquared_adj,
        full_model.fvalue,
        full_model.f_pvalue,
        full_model.aic,
        full_model.bic,
        int(full_model.nobs),
        int(full_model.df_model),
        int(full_model.df_resid)
    ]
})
print(full_model_summary.round(4))

# Fit the reduced model (DD removed)
reduced_model = smf.ols(
    formula='''
Q("SALE PRICE") ~
Q("RESIDENTIAL UNITS") +
Q("COMMERCIAL UNITS") +
Q("LAND SQUARE FEET") +
Q("GROSS SQUARE FEET") +
Q("YEAR BUILT") +
MM +
Q("BOROUGH_Brooklyn") +
Q("BOROUGH_Manhattan") +
Q("BOROUGH_Queens") +
Q("BOROUGH_Staten Island") +
Q("BUILDING CLASS CATEGORY_2") +
Q("BUILDING CLASS CATEGORY_3")
''',
    data=NYC_housing
).fit()

reduced_model_summary = pd.DataFrame({
    "Statistic": [
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "Prob (F-statistic)",
        "AIC",
        "BIC"
    ],
    "Value": [
        reduced_model.rsquared,
        reduced_model.rsquared_adj,
        reduced_model.fvalue,
        reduced_model.f_pvalue,
        reduced_model.aic,
        reduced_model.bic
    ]
}).round(4)

print(reduced_model_summary)

reduced_model_coefficients = pd.DataFrame({
    "Coefficient": reduced_model.params.round(4),
    "Standard Error": reduced_model.bse.round(4),
    "t-value": reduced_model.tvalues.round(4),
    "p-value": reduced_model.pvalues,
    "95% CI Lower": reduced_model.conf_int()[0].round(4),
    "95% CI Upper": reduced_model.conf_int()[1].round(4)
})
reduced_model_coefficients["p-value"] = reduced_model_coefficients["p-value"].map(lambda x: f"{x:.3e}")
print(reduced_model_coefficients)

# Fit the simple model
simple_model = smf.ols(
    formula='''
Q("SALE PRICE") ~
Q("GROSS SQUARE FEET") +
Q("LAND SQUARE FEET") +
Q("YEAR BUILT") +
MM
''',
    data=NYC_housing
).fit()

simple_model_summary = pd.DataFrame({
    "Statistic": [
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "Prob (F-statistic)",
        "AIC",
        "BIC"
    ],
    "Value": [
        simple_model.rsquared,
        simple_model.rsquared_adj,
        simple_model.fvalue,
        simple_model.f_pvalue,
        simple_model.aic,
        simple_model.bic
    ]
}).round(4)

print(simple_model_summary)

simple_model_coefficients = pd.DataFrame({
    "Coefficient": simple_model.params.round(4),
    "Standard Error": simple_model.bse.round(4),
    "t-value": simple_model.tvalues.round(4),
    "p-value": simple_model.pvalues
})

simple_model_coefficients["p-value"] = (
    simple_model_coefficients["p-value"]
    .apply(lambda x: "<0.001" if x < 0.001 else f"{x:.4f}")
)

print(simple_model_coefficients)

# Find the final model
final_model = reduced_model

# OLS Assumptions

# The residuals have zero mean.
residual_mean = pd.DataFrame({
    "Statistic": ["Mean of residuals"],
    "Value": [final_model.resid.mean()]
})

print(residual_mean) # Residuals have zero mean

# Obtain the residuals from the final OLS model
residuals = final_model.resid

# Calculate the mean of the residuals
mean_residual = residuals.mean()

print("Mean of residuals:", mean_residual)

# Check whether the mean is numerically close to zero
if np.isclose(mean_residual, 0, atol=1e-6):
    print("The mean of the residuals is effectively zero.")
else:
    print("The mean of the residuals is not effectively zero.")

# The variance of the residuals is constant (and finite) --> Homoskedasticity
# Breusch - Pagan Test (for heteroskedasticity)
bp_test = het_breuschpagan(
    final_model.resid,
    final_model.model.exog
)

final_model_breusch_pagan = pd.DataFrame({
    "Statistic": [
        "LM Statistic",
        "LM p-value",
        "F Statistic",
        "F p-value"
    ],
    "Value": bp_test
})

print(final_model_breusch_pagan.round(4)) # Heteroskedasticity

# The residuals are linearly independent of one another
# Durbin-Watson test (for autocorrelation)
final_model_durbin_watson = pd.DataFrame({
    "Statistic": ["Durbin-Watson"],
    "Value": [durbin_watson(final_model.resid)]
})

print(final_model_durbin_watson.round(4)) # DW = 1.139

# There is no relationship between the residuals and each of explanatory variables
# Measure for (linear) dependence e.g. linear correlation
# Correlation between residuals and each explanatory variable
explanatory_variables = final_model.model.exog_names

residual_correlations = pd.Series(
    {
        variable: np.corrcoef(
            final_model.resid,
            final_model.model.exog[:, i]
        )[0, 1]
        for i, variable in enumerate(explanatory_variables)
        if variable != "Intercept"
    },
    name="Correlation with residuals"
)

print(residual_correlations.round(4))

# Check for systematic relationships between residuals and fitted values
residuals = final_model.resid
fitted_values = final_model.fittedvalues

plt.figure(figsize=(8, 5))
plt.scatter(fitted_values, residuals, alpha=0.3)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals versus fitted values")
plt.tight_layout()
plt.show()


# Check for systematic relationships between residuals
# and numerical explanatory variables
model_data = final_model.model.data.frame
residuals = final_model.resid

variable_labels = {
    "RESIDENTIAL UNITS": "Residential units",
    "COMMERCIAL UNITS": "Commercial units",
    "LAND SQUARE FEET": "Land square feet",
    "GROSS SQUARE FEET": "Gross square feet",
    "YEAR BUILT": "Year built",
    "MM": "Sale month",
    "DD": "Sale day"
}

for variable, label in variable_labels.items():
    plt.figure(figsize=(10, 7))

    plt.scatter(
        model_data[variable],
        residuals,
        alpha=0.3
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel(label, fontsize=14, fontweight="normal")
    plt.ylabel("Residuals", fontsize=14, fontweight="normal")
    plt.title(
        f"Residuals versus {label.lower()}",
        fontsize=16,
        fontweight="normal"
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()


# The residuals are normally distributed
# Jarque-Bera test (for normality of the distribution)
jb = jarque_bera(final_model.resid)

final_model_jarque_bera = pd.DataFrame({
    "Statistic": [
        "Jarque-Bera",
        "JB p-value",
        "Skewness",
        "Kurtosis"
    ],
    "Value": jb
})

print(final_model_jarque_bera.round(4))

# Histogram of residuals
plt.figure(figsize=(6,5))

plt.hist(
    final_model.resid,
    bins=40,
    edgecolor="black"
)

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of residuals")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

conf_int = final_model.conf_int()

final_model_coefficients = pd.DataFrame({
    "Coefficient": final_model.params.round(2),
    "Standard Error": final_model.bse.round(2),
    "t-value": final_model.tvalues.round(2),
    "p-value": final_model.pvalues,
    "95% CI": (
        "[" + conf_int[0].round(2).astype(str)
        + ", " + conf_int[1].round(2).astype(str) + "]"
    )
})

final_model_coefficients["p-value"] = (
    final_model_coefficients["p-value"]
    .apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")
)

print(final_model_coefficients.to_string())

final_model_summary = pd.DataFrame({
    "Statistic": [
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "Prob (F-statistic)",
        "AIC",
        "BIC"
    ],
    "Value": [
        final_model.rsquared,
        final_model.rsquared_adj,
        final_model.fvalue,
        final_model.f_pvalue,
        final_model.aic,
        final_model.bic
    ]
}).round(4)

#####--------------------NYC Housing Sales for SimDec -------------------------
# Read the NYC Calendar Sales 2015 dataset
NYC_housing_data = pd.read_csv("NYCHousing2015.csv")

# Convert numerical variables to numeric format
numeric_variables = [
    "RESIDENTIAL UNITS",
    "COMMERCIAL UNITS",
    "LAND SQUARE FEET",
    "GROSS SQUARE FEET",
    "YEAR BUILT",
    "SALE PRICE"
    ]

for variable in numeric_variables:
    NYC_housing_data[variable] = pd.to_numeric(NYC_housing_data[variable], errors="coerce")
    
# Calculate the median and MAD of SALE_PRICE
sale_price = NYC_housing_data["SALE PRICE"]
median_price = sale_price.median()
mad = np.median(np.abs(sale_price - median_price))

# Scale MAD to make it comparable with standard deviation
scaled_mad = mad / 0.6744897501960817

# Define the outlier limits
lower_limit = median_price - 3 * scaled_mad
upper_limit = median_price + 3 * scaled_mad

# Identify SALE_PRICE outliers
price_outliers = ((sale_price < lower_limit) | (sale_price > upper_limit))

# Remove SALE_PRICE outliers
NYC_housing_data = NYC_housing_data.loc[~price_outliers].copy()

# Calculate the average of sale price for each borough
borough_mean_sale_price = (NYC_housing_data.groupby("BOROUGH")["SALE PRICE"].mean().sort_values())

print(borough_mean_sale_price)

# Convert BOROUGH names to numerical category codes
borough_codes = {
    "BRONX": 1,
    "STATEN ISLAND": 2,
    "QUEENS": 3,
    "BROOKLYN": 4,
    "MANHATTAN": 5
}

NYC_housing_data["BOROUGH"] = (NYC_housing_data["BOROUGH"].astype("string").
                               str.strip().str.upper().map(borough_codes).astype("Int64"))

# Clean BUILDING CLASS CATEGORY names
NYC_housing_data["BUILDING CLASS CATEGORY"] = (NYC_housing_data["BUILDING CLASS CATEGORY"]
                                               .astype("string").str.strip())

# Retain building class categories 01–03
NYC_housing_data = NYC_housing_data.loc[
    NYC_housing_data["BUILDING CLASS CATEGORY"]
    .str.extract(r"^(\d+)", expand=False)
    .astype("Int64")
    .isin([1, 2, 3])
].copy()

# Calculate the average sale price for each building class category
building_class_mean_sale_price = (NYC_housing_data.groupby("BUILDING CLASS CATEGORY")["SALE PRICE"].mean().sort_values())

print(building_class_mean_sale_price)

# Assign numerical codes from low to high mean sale price
building_class_codes = {
    category: code
    for code, category in enumerate(
        building_class_mean_sale_price.index,
        start=1
    )
}

print(building_class_codes)

# Convert category names to numerical codes
NYC_housing_data["BUILDING CLASS CATEGORY"] = (
    NYC_housing_data["BUILDING CLASS CATEGORY"]
    .map(building_class_codes)
    .astype("Int64")
)

# Convert SALE_DATE to datetime
NYC_housing_data["SALE DATE"] = pd.to_datetime(
    NYC_housing_data["SALE DATE"],
    errors="coerce"
)

# Select numerical input variables
numerical_inputs_nyc = [
    "RESIDENTIAL UNITS",
    "COMMERCIAL UNITS",
    "LAND SQUARE FEET",
    "GROSS SQUARE FEET",
    "YEAR BUILT"
    ]

# Calculate correlations
input_correlation_nyc = NYC_housing_data[numerical_inputs_nyc].corr()

# Plot correlation heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    input_correlation_nyc,
    annot=True,
    annot_kws={"size": 12}, 
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)
#plt.title("(a) Correlation matrix of numerical inputs in the New York City housing sales dataset",fontsize=14,pad=15)
plt.tight_layout()
plt.show()

# Sale price distribution
plt.figure(figsize=(9, 6))
plt.hist(NYC_housing_data["SALE PRICE"], bins=41, weights=(np.ones(len(NYC_housing_data)) / len(NYC_housing_data)), edgecolor="black")
plt.xlabel("Sale Price")
plt.ylabel("Probability")
plt.title("(a) Distribution of sale price in the New York City housing sales dataset", fontsize=14, pad=15)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Save the preprocessed dataset for SimDec analysis
NYC_housing_data.to_csv("simdec_nyc_housing2015.csv", index=False)









