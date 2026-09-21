import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns


# Read Iris dataset
iris = pd.read_csv("iris_data.csv")

# Select numerical variables for the scatterplot matrix
measurement_variables = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

g = sns.pairplot(
    iris,
    vars=measurement_variables,
    hue="class",
    diag_kind="hist",
    plot_kws={"alpha": 0.7, "s": 35},
    diag_kws={"bins": 15, "alpha": 0.6}
)

# Increase axis-label and tick-label sizes
for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=13)
        ax.set_ylabel(ax.get_ylabel(), fontsize=13)
        ax.tick_params(axis="both", labelsize=11)

# Adjust the legend and place it at the upper-right of the entire figure
if g._legend is not None:
    g._legend.set_title("Class", prop={"size": 14})

    for text in g._legend.texts:
        text.set_fontsize(12)

    g._legend.set_loc("upper left")
    g._legend.set_bbox_to_anchor((0.85, 1.00))

# Leave space on the right for the legend
g.figure.subplots_adjust(right=0.84, top=0.93)
plt.show()

# Check the basic information about the dataset
print(iris.head())
print(iris.shape)
print(iris.dtypes)
print(iris.isnull().sum())
print(iris.columns)
print(iris["class"].value_counts())
print(iris.describe().round(2))

# Plot distributions of the four nummerical vairables
numerical_variables =[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
    ]

# Numerical measurement variables and display names
variable_labels = {
    "sepal_length": "Sepal Length",
    "sepal_width": "Sepal Width",
    "petal_length": "Petal Length",
    "petal_width": "Petal Width"
}

# Data preparation
# Separate the four variables for clustering
X = iris[numerical_variables]

# Retain the known species labels for later evaluation
true_class = iris["class"]

# Standardize the four clustering variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Standardize data
X_scaled = pd.DataFrame(X_scaled, columns = numerical_variables, index = iris.index)
print(X_scaled.head())
print(X_scaled.describe().round(2))

# Elbow method to find the optimal number of clusters
# Elbow method: k = 1 to 10
inertia_values = []
elbow_k_values = range(1, 11)

for k in elbow_k_values:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(elbow_k_values, inertia_values, marker="o")
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squared Euclidean distances")
plt.title("Elbow Method for finding number of clusters")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final k-means model with k = 3
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10,random_state=42)
iris["cluster"] = kmeans.fit_predict(X_scaled)

# Check cluster sizes
print("\nCluster sizes:")
print(iris["cluster"].value_counts().sort_index())
cluster_sizes = (
    iris["cluster"]
    .value_counts()
    .sort_index()
    .rename("Number of observations")
    .to_frame()
    )
cluster_sizes.index.name = "Cluster"
print(cluster_sizes)

# Convert cluster centres back to the original measurement scale
cluster_centres = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centres = pd.DataFrame(cluster_centres, columns=numerical_variables)
cluster_centres.index.name = "cluster"
print("\nCluster centres in original measurement units:")
print(cluster_centres.round(2))

# Create a table for cluster size and centres
cluster_summary = cluster_centres.round(2).copy()
cluster_summary.insert(0, "Number of observations", iris["cluster"].value_counts().sort_index())

cluster_summary.index.name = "Cluster"

print("\nCluster sizes and centres:")
print(cluster_summary)

# Compare clusters with the known species labels
cluster_species_table = pd.crosstab(iris["cluster"], true_class, rownames=["K-means cluster"], colnames=["True class"]    )
print("\nClusters compared with known species:")
print(cluster_species_table)

# Measure agreement between the two groupings
ari = adjusted_rand_score(true_class, iris["cluster"])
print("\nAdjusted Rand Index:")
print(round(ari, 3))

silhouette = silhouette_score(X_scaled,iris["cluster"])
print("\nSilhouette score:")
print(round(silhouette, 3))

# Reduce the four standardized variables to two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Red, blue and green
cluster_colors = ListedColormap(["red", "blue", "green"])

# Compare the ture class and k-means clusters in PCA space
species_names = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
    ]

species_codes = pd.Categorical(true_class, categories = species_names).codes

# Use different colours for true classes 
species_colors = ListedColormap(["purple", "orange", "darkcyan"])

fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
# (a) True class
true_scatter = axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=species_codes,
    cmap=species_colors,
    alpha=0.8
    )

axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].set_title("(a) True class")
axes[0].grid(True)

true_handles = true_scatter.legend_elements()[0]
axes[0].legend(
    true_handles,
    species_names,
    title="Species"
    )

# (b) K-means clusters
cluster_scatter = axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=iris["cluster"],
    cmap=cluster_colors,
    alpha=0.8
    )

axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].set_title("(b) k-means Clusters")
axes[1].grid(True)

axes[1].legend(
    *cluster_scatter.legend_elements(),
    title="Cluster"
    )
plt.tight_layout()
plt.show()

####----------------------Iris for SimDec -------------------------------------
# Read Iris dataset
iris = pd.read_csv("iris_data.csv")

# Numerical input vairbales
numerical_inputs_iris = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

# Calculate correlations between numerical inputs
input_correlation_iris =(iris[numerical_inputs_iris].corr()) 

# Plot correlation heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(input_correlation_iris, annot = True, fmt=".2f", 
            annot_kws={"size": 14}, 
            cmap="coolwarm", vmin=-1, vmax=1, square=True)

#plt.title("(c) Correlation matrix of numerical inputs in the bank marketing dataset", fontsize=16,pad=15)
plt.xticks(rotation=45, ha="right", fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.show()

# Convert class into numerical variable
class_mapping = {
    "Iris-setosa": 1,
    "Iris-versicolor": 2,
    "Iris-virginica": 3
}

iris["class"] = (
    iris["class"]
    .astype("string")
    .str.strip()
    .map(class_mapping)
    .astype("Int64")
)

# Class distribution
plt.figure(figsize=(9, 6))
plt.hist(iris["class"], bins=11, weights=np.ones(len(iris)) / len(iris), edgecolor="black")
plt.xticks(
    [1, 2, 3],
    [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ]
)
plt.xlabel("Class")
plt.ylabel("Probalitity")
plt.title("(c) Distribution of the class in the iris dataset", fontsize=14, pad=15)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()   

iris.to_csv("simdec_iris.csv", index=False)
