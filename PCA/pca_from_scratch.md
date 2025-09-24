# PCA from scratch

Principal component analysis (PCA) is a dimensionality reduction technique that transforms a data set into a set of orthogonal components — called principal components — which capture the maximum variance in the data. PCA simplifies complex data sets while preserving their most important structures.

# Why it matters?

Let’s say you are trying to predict house prices and you have 10 columns, like square footage, number of rooms, distance from the city, and so on.

Some of these columns may be giving you the same kind of information. PCA finds patterns in these columns and builds new axes to represent the data more efficiently.

These new axes are called principal components. Each one shows how much of the total variation in the data it can explain.

This video does an excellent job of explaining PCA in very simple terms.

[PCA Visually Explained](https://youtu.be/FD4DeN81ODY?si=6wYW_4rYxHdPWpWr)

# Dataset

For the following walkthrough, we will be working with the famous “Iris” dataset that has been deposited on the UCI machine learning repository
(https://archive.ics.uci.edu/ml/datasets/Iris).

The iris dataset contains measurements for 150 iris flowers from three different species.

The three classes in the Iris dataset are:

Iris-setosa (n=50)
Iris-versicolor (n=50)
Iris-virginica (n=50)
And the four features of in Iris dataset are:

sepal length in cm
sepal width in cm
petal length in cm
petal width in cm


## Loading Dataset

In order to load the Iris data directly from the UCI repository, we are going to use the superb pandas library.


```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler

```


```python
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_len</th>
      <th>sepal_wid</th>
      <th>petal_len</th>
      <th>petal_wid</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>




```python
# split data table into data X and class labels y

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
```


```python
# Feature columns and names
feature_cols = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
feature_names = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']

# Color mapping for classes
color_discrete_map = {
    'Iris-setosa': '#FF6B6B',
    'Iris-versicolor': '#4ECDC4', 
    'Iris-virginica': '#45B7D1'
}

# 1. SCATTER PLOT MATRIX
print("Creating Scatter Plot Matrix...")

# Calculate number of unique pairs
n_features = len(feature_cols)
pairs = list(combinations(range(n_features), 2))
n_pairs = len(pairs)

# Create subplot grid (3x2 for 6 pairs)
rows = 2
cols = 3

fig_scatter = make_subplots(
    rows=rows, 
    cols=cols,
    # subplot_titles=[f"{feature_names[pair[1]]} vs {feature_names[pair[0]]}" for pair in pairs],
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

# Add scatter plots for each pair
for idx, (i, j) in enumerate(pairs):
    row = (idx // cols) + 1
    col = (idx % cols) + 1
    
    for class_name, color in color_discrete_map.items():
        class_data = df[df['class'] == class_name]
        
        fig_scatter.add_trace(
            go.Scatter(
                x=class_data[feature_cols[i]],
                y=class_data[feature_cols[j]],
                mode='markers',
                name=class_name,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                showlegend=(idx == 0),  # Only show legend once
                legendgroup=class_name
            ),
            row=row, col=col
        )
    
    # Update axes for this subplot
    fig_scatter.update_xaxes(
        title_text=feature_names[i], 
        row=row, col=col,
        showgrid=True,
        gridcolor='lightgray'
    )
    fig_scatter.update_yaxes(
        title_text=feature_names[j], 
        row=row, col=col,
        showgrid=True,
        gridcolor='lightgray'
    )

# Update scatter plot layout
fig_scatter.update_layout(
    title={
        'text': "Iris Dataset - Feature Relationships (Scatter Plot Matrix)",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    height=600,
    width=1000,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='white'
)

fig_scatter.show()

# 2. FEATURE DISTRIBUTIONS (HISTOGRAMS)
print("Creating Feature Distribution Plots...")

fig_hist = make_subplots(
    rows=2, 
    cols=2,
    # subplot_titles=feature_names,
    vertical_spacing=0.15,
    horizontal_spacing=0.15
)

for idx, (col_name, feature_name) in enumerate(zip(feature_cols, feature_names)):
    row = (idx // 2) + 1
    col = (idx % 2) + 1
    
    for class_name, color in color_discrete_map.items():
        class_data = df[df['class'] == class_name]
        
        fig_hist.add_trace(
            go.Histogram(
                x=class_data[col_name],
                name=class_name,
                marker_color=color,
                opacity=0.7,
                nbinsx=15,
                showlegend=(idx == 0),  # Only show legend once
                legendgroup=class_name
            ),
            row=row, col=col
        )
    
    # Update axes for this subplot
    fig_hist.update_xaxes(
        title_text=feature_name, 
        row=row, col=col,
        showgrid=True,
        gridcolor='lightgray'
    )
    fig_hist.update_yaxes(
        title_text="Count", 
        row=row, col=col,
        showgrid=True,
        gridcolor='lightgray'
    )

# Update histogram layout
fig_hist.update_layout(
    title={
        'text': "Iris Dataset - Feature Distributions",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    height=600,
    width=800,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='white',
    barmode='overlay'  # Overlay histograms for better comparison
)

fig_hist.show()
```

![PCA](https://github.com/ish-codes-magic/ML-algorithms-from-scratch/blob/main/img/pca_exploratory.png)
![PCA](https://github.com/ish-codes-magic/ML-algorithms-from-scratch/blob/main/img/pca_exploratory_2.png)
    



# How PCA works?

PCA uses linear algebra to transform data into new features called principal components. It finds these by calculating eigenvectors (directions) and eigenvalues (importance) from the covariance matrix. PCA selects the top components with the highest eigenvalues and projects the data onto them simplify the dataset.



## Step 1: Standardize the Data

Different features may have different units and scales like salary vs. age. To compare them fairly PCA first standardizes the data by making each feature have:

* A mean of 0
* A standard deviation of 1

$Z = \frac{X - \mu}{\sigma}$

where:

* μ is the mean of independent features μ ={μ1,μ2,⋯,μm}
* σ is the standard deviation of independent features σ ={σ1,σ2,⋯,σm}


```python
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
```


```python
print("Before Standardization:")
print(X[:5])
print("After Standardization:")
print(X_std[:5])
```

    Before Standardization:
    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]
    After Standardization:
    [[-0.90068117  1.03205722 -1.3412724  -1.31297673]
     [-1.14301691 -0.1249576  -1.3412724  -1.31297673]
     [-1.38535265  0.33784833 -1.39813811 -1.31297673]
     [-1.50652052  0.10644536 -1.2844067  -1.31297673]
     [-1.02184904  1.26346019 -1.3412724  -1.31297673]]
    


```python
print("The mean of each feature of the X dataset is:")
print(scaler.mean_)
print("The standard deviation of each feature of the X dataset is:")
print(scaler.scale_)

```

    The mean of each feature of the X dataset is:
    [5.84333333 3.054      3.75866667 1.19866667]
    The standard deviation of each feature of the X dataset is:
    [0.82530129 0.43214658 1.75852918 0.76061262]
    

## Step 2: Calculate Covariance Matrix

Next PCA calculates the covariance matrix to see how features relate to each other whether they increase or decrease together. The covariance between two features x1 and x2 is:

$ cov(x_1, x_2) = \frac{\sum_{i=1}^{n} (x1_i - \tilde{x1})(x2_i - \tilde{x2})}{n-1} $

Where:

* $\tilde{x1}$ and $\tilde{x2}$ are the mean values of features x1 and x2
* n is the number of data points

The value of covariance can be positive, negative or zeros.

In terms of matrix, it can be represented as:

$ cov = \frac{(X - \bar{x})^{T}(X - \bar{x})}{n-1} $


```python
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print(f'Covariance matrix \n{cov_mat}')
```

    Covariance matrix 
    [[ 1.00671141 -0.11010327  0.87760486  0.82344326]
     [-0.11010327  1.00671141 -0.42333835 -0.358937  ]
     [ 0.87760486 -0.42333835  1.00671141  0.96921855]
     [ 0.82344326 -0.358937    0.96921855  1.00671141]]
    

There is a direct implementation of calculating covariance in Numpy using the `cov` function.


```python
cov_mat = np.cov(X_std.T)
print(f'NumPy covariance matrix: \n{cov_mat}')
```

    NumPy covariance matrix: 
    [[ 1.00671141 -0.11010327  0.87760486  0.82344326]
     [-0.11010327  1.00671141 -0.42333835 -0.358937  ]
     [ 0.87760486 -0.42333835  1.00671141  0.96921855]
     [ 0.82344326 -0.358937    0.96921855  1.00671141]]
    

## Step 3: Find the Principal Components

PCA identifies new axes where the data spreads out the most:

* 1st Principal Component (PC1): The direction of maximum variance (most spread).
* 2nd Principal Component (PC2): The next best direction, perpendicular to PC1 and so on.


These directions come from the eigenvectors of the covariance matrix and their importance is measured by eigenvalues. For a square matrix A an eigenvector X (a non-zero vector) and its corresponding eigenvalue λ satisfy:

$AX = \lambda X$

This means:

* When A acts on X it only stretches or shrinks X by the scalar λ.
* The direction of X remains unchanged hence eigenvectors define "stable directions" of A.

Eigenvalues help rank these directions by importance.


```python
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print(f'Eigenvectors \n{eig_vecs}')
print(f'\nEigenvalues \n{eig_vals}')
```

    Eigenvectors 
    [[ 0.52237162 -0.37231836 -0.72101681  0.26199559]
     [-0.26335492 -0.92555649  0.24203288 -0.12413481]
     [ 0.58125401 -0.02109478  0.14089226 -0.80115427]
     [ 0.56561105 -0.06541577  0.6338014   0.52354627]]
    
    Eigenvalues 
    [2.93035378 0.92740362 0.14834223 0.02074601]
    

While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve, most PCA implementations perform a Singular Value Decomposition (SVD).

There are two main reasons why SVD is often preferred over directly calculating eigenvalues and eigenvectors from the covariance matrix:

1. **Numerical Stability**: SVD is numerically more stable and less prone to errors when handling large or ill-conditioned matrices.
2. **Efficiency**: Computing SVD is often faster, especially for large datasets. It provides an elegant way to perform PCA without explicitly calculating the covariance matrix.

To have a more deeper understanding of how SVD works, follow this article: [SVD](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)


```python
# Correct SVD approach for PCA
# Apply SVD to the centered data matrix (not its transpose)
U, S, Vt = np.linalg.svd(X_std, full_matrices=False)

# The relationship between SVD and eigendecomposition:
# - U contains the left singular vectors
# - S contains the singular values 
# - Vt contains the right singular vectors (which are our principal components)
# - Eigenvalues = (singular_values^2) / (n-1)

print("SVD Components:")
print(f"U shape: {U.shape} (data projections onto principal components)")
print(f"S shape: {S.shape} (singular values)")
print(f"Vt shape: {Vt.shape} (principal components as rows)")

# Calculate eigenvalues from singular values
eigenvalues_from_svd = (S**2) / (X_std.shape[0] - 1)
print(f"\nEigenvalues from SVD: {eigenvalues_from_svd}")
print(f"Eigenvalues from covariance matrix: {eig_vals}")
```

    SVD Components:
    U shape: (150, 4) (data projections onto principal components)
    S shape: (4,) (singular values)
    Vt shape: (4, 4) (principal components as rows)
    
    Eigenvalues from SVD: [2.93035378 0.92740362 0.14834223 0.02074601]
    Eigenvalues from covariance matrix: [2.93035378 0.92740362 0.14834223 0.02074601]
    

## Step 4: Pick the Top Directions & Transform Data

After calculating the eigenvalues and eigenvectors PCA ranks them by the amount of information they capture. We then:

* Select the top k components hat capture most of the variance like 95%.
* Transform the original dataset by projecting it onto these top components.

This means we reduce the number of features (dimensions) while keeping the important patterns in the data.

![PCA](https://github.com/ish-codes-magic/ML-algorithms-from-scratch/blob/main/img/Principal-Componenent-Analysisi.webp)


In the above image the original dataset has two features "Radius" and "Area" represented by the black axes. PCA identifies two new directions: PC₁ and PC₂ which are the principal components.

* These new axes are rotated versions of the original ones. PC₁ captures the maximum variance in the data meaning it holds the most information while PC₂ captures the remaining variance and is perpendicular to PC₁.

* The spread of data is much wider along PC₁ than along PC₂. This is why PC₁ is chosen for dimensionality reduction. By projecting the data points (blue crosses) onto PC₁ we effectively transform the 2D data into 1D and retain most of the important structure and patterns.

The common approach is to rank the eigenvalues from highest to lowest in order choose the top k eigenvectors.


```python
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
print('Eigenvectors in descending order of their eigenvalues:')
for i in eig_pairs:
    print(i[1])

```

    Eigenvalues in descending order:
    2.9303537755893165
    0.9274036215173417
    0.14834222648163994
    0.020746013995596356
    Eigenvectors in descending order of their eigenvalues:
    [ 0.52237162 -0.26335492  0.58125401  0.56561105]
    [-0.37231836 -0.92555649 -0.02109478 -0.06541577]
    [-0.72101681  0.24203288  0.14089226  0.6338014 ]
    [ 0.26199559 -0.12413481 -0.80115427  0.52354627]
    

First k=2 most relevant Principal components are:


```python
k = 2
print('The first', k, 'principal components are:')
for i in range(k):
    print(eig_pairs[i][1])
```

    The first 2 principal components are:
    [ 0.52237162 -0.26335492  0.58125401  0.56561105]
    [-0.37231836 -0.92555649 -0.02109478 -0.06541577]
    

Incase of SVD, the same principal components can be computed using the first k components of $V^{T}$ vector.


```python
# Correct way to extract principal components from SVD
# The principal components are the rows of Vt (right singular vectors)
principal_components_svd = Vt[:k]

print(f"Shape of principal components from SVD: {principal_components_svd.shape}")
print("Each row is a principal component with {0} features".format(principal_components_svd.shape[1]))
```

    Shape of principal components from SVD: (2, 4)
    Each row is a principal component with 4 features
    


```python
print('The first', k, 'principal components using SVD are:')
for i in range(k):
    print(principal_components_svd[i])

```

    The first 2 principal components using SVD are:
    [ 0.52237162 -0.26335492  0.58125401  0.56561105]
    [-0.37231836 -0.92555649 -0.02109478 -0.06541577]
    


```python
print('Principal components from eigendecomposition:')
for i in range(k):
    print(f"PC{i+1}: {eig_pairs[i][1]}")
    
print('\nPrincipal components from SVD:')
for i in range(k):
    # Note: SVD might have different signs, so we compare absolute values
    pc_svd = principal_components_svd[i]
        
    print(f"PC{i+1}: {pc_svd}")
```

    Principal components from eigendecomposition:
    PC1: [ 0.52237162 -0.26335492  0.58125401  0.56561105]
    PC2: [-0.37231836 -0.92555649 -0.02109478 -0.06541577]
    
    Principal components from SVD:
    PC1: [ 0.52237162 -0.26335492  0.58125401  0.56561105]
    PC2: [-0.37231836 -0.92555649 -0.02109478 -0.06541577]
    

## Step 5: Variance

After sorting the eigenpairs, the next question is “how many principal components are we going to choose for our new feature subspace?” A useful measure is the so-called “explained variance,” which can be calculated from the eigenvalues. The explained variance tells us how much information (variance) can be attributed to each of the principal components.


```python
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
```

The similar implementation using SVD to get the variance is as follows.


```python
# Calculate explained variance from SVD approach
# Eigenvalues from SVD are already calculated: eigenvalues_from_svd = (S**2) / (n-1)
tot_svd = sum(eigenvalues_from_svd)
var_exp_svd = [(i / tot_svd)*100 for i in sorted(eigenvalues_from_svd, reverse=True)]
cum_var_exp_svd = np.cumsum(var_exp_svd)

```


```python
# Create Plotly visualization for explained variance
fig_variance = go.Figure()

# Component labels
component_labels = [f'PC{i+1}' for i in range(4)]

# Add bar chart for individual explained variance (Eigendecomposition)
fig_variance.add_trace(go.Bar(
    x=component_labels,
    y=var_exp,
    name='Individual Explained Variance (Eigen)',
    marker_color='rgba(55, 128, 191, 0.7)',
    yaxis='y1'
))

# Add line chart for cumulative explained variance (Eigendecomposition)
fig_variance.add_trace(go.Scatter(
    x=component_labels,
    y=cum_var_exp,
    mode='lines+markers',
    name='Cumulative Explained Variance (Eigen)',
    line=dict(color='rgba(219, 64, 82, 1.0)', width=3),
    marker=dict(size=8),
    yaxis='y2'
))

# Add bar chart for individual explained variance (SVD)
fig_variance.add_trace(go.Bar(
    x=component_labels,
    y=var_exp_svd,
    name='Individual Explained Variance (SVD)',
    marker_color='rgba(50, 171, 96, 0.7)',
    yaxis='y1',
    opacity=0.8
))

# Add line chart for cumulative explained variance (SVD)
fig_variance.add_trace(go.Scatter(
    x=component_labels,
    y=cum_var_exp_svd,
    mode='lines+markers',
    name='Cumulative Explained Variance (SVD)',
    line=dict(color='rgba(128, 0, 128, 1.0)', width=3, dash='dash'),
    marker=dict(size=8),
    yaxis='y2'
))

# Update layout with dual y-axes
fig_variance.update_layout(
    title={
        'text': 'Explained Variance Analysis: Eigendecomposition vs SVD',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis=dict(
        title='Principal Components',
        # titlefont=dict(size=14),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='Individual Explained Variance (%)',
        # titlefont=dict(size=14, color='rgb(55, 128, 191)'),
        tickfont=dict(size=12, color='rgb(55, 128, 191)'),
        side='left'
    ),
    yaxis2=dict(
        title='Cumulative Explained Variance (%)',
        # titlefont=dict(size=14, color='rgb(219, 64, 82)'),
        tickfont=dict(size=12, color='rgb(219, 64, 82)'),
        side='right',
        overlaying='y',
        range=[0, 105]
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    plot_bgcolor='white',
    width=900,
    height=600,
    showlegend=True
)

# Add grid lines
fig_variance.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_variance.update_yaxes(showgrid=True, gridcolor='lightgray')

fig_variance.show()
```

![PCA](https://github.com/ish-codes-magic/ML-algorithms-from-scratch/blob/main/img/pca_exploratory_3.png)

## Step 6: Project the data

We first create the projection matrix. It is basically just a matrix of our concatenated top k eigenvectors.


Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace, by choosing the “top 2” eigenvectors with the highest eigenvalues to construct our d×k-dimensional eigenvector matrix W.


```python
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('The projection matrix W using eigendecomposition:')
print(matrix_w)
```

    The projection matrix W using eigendecomposition:
    [[ 0.52237162 -0.37231836]
     [-0.26335492 -0.92555649]
     [ 0.58125401 -0.02109478]
     [ 0.56561105 -0.06541577]]
    


```python
matrix_w = np.hstack((principal_components_svd[0].reshape(4,1),
                      principal_components_svd[1].reshape(4,1)))
print('The projection matrix W using SVD:')
print(matrix_w)
```

    The projection matrix W using SVD:
    [[ 0.52237162 -0.37231836]
     [-0.26335492 -0.92555649]
     [ 0.58125401 -0.02109478]
     [ 0.56561105 -0.06541577]]
    

Finally, we can transform the data X via the projection matrix W to obtain a k-dimensional feature subspace.




```python
Y = X_std.dot(matrix_w)
```

Next, we create some visualisations to visualise our data in the principal components space


```python
# Convert matplotlib PCA scatter plot to Plotly
fig_pca = go.Figure()

# Define colors and labels
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green (Plotly default colors)
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Add scatter plots for each class
for lab, color in zip(labels, colors):
    class_mask = (y == lab)
    fig_pca.add_trace(go.Scatter(
        x=Y[class_mask, 0],
        y=Y[class_mask, 1],
        mode='markers',
        name=lab,
        marker=dict(
            color=color,
            size=8,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        hovertemplate=
        '<b>%{fullData.name}</b><br>' +
        'PC1: %{x:.3f}<br>' +
        'PC2: %{y:.3f}<br>' +
        '<extra></extra>'
    ))

# Update layout
fig_pca.update_layout(
    title={
        'text': 'PCA Visualization: Iris Dataset in Principal Component Space',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis=dict(
        title='Principal Component 1',
        # titlefont=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=1
    ),
    yaxis=dict(
        title='Principal Component 2',
        # titlefont=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=1
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.1,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    plot_bgcolor='white',
    width=800,
    height=600,
    showlegend=True
)

fig_pca.show()

# Optional: Add statistical information
print("PCA Transformation Summary:")
print(f"Data shape after PCA: {Y.shape}")
print(f"Original data shape: {X_std.shape}")
print(f"Dimensionality reduction: {X_std.shape[1]} → {Y.shape[1]} features")

# Calculate and display class separation metrics
from scipy.spatial.distance import pdist, squareform

print("\nClass Separation Analysis in PCA Space:")
for i, lab in enumerate(labels):
    class_data = Y[y == lab]
    centroid = np.mean(class_data, axis=0)
    print(f"{lab} centroid: PC1={centroid[0]:.3f}, PC2={centroid[1]:.3f}")
    
# Calculate between-class distances
centroids = []
for lab in labels:
    class_data = Y[y == lab]
    centroids.append(np.mean(class_data, axis=0))

centroids = np.array(centroids)
distances = pdist(centroids)
print(f"\nBetween-class distances in PCA space:")
print(f"Setosa ↔ Versicolor: {distances[0]:.3f}")
print(f"Setosa ↔ Virginica: {distances[1]:.3f}")
print(f"Versicolor ↔ Virginica: {distances[2]:.3f}")
```

![PCA](https://github.com/ish-codes-magic/ML-algorithms-from-scratch/blob/main/img/pca_exploratory_4.png)

    PCA Transformation Summary:
    Data shape after PCA: (150, 2)
    Original data shape: (150, 4)
    Dimensionality reduction: 4 → 2 features
    
    Class Separation Analysis in PCA Space:
    Iris-setosa centroid: PC1=-2.220, PC2=-0.292
    Iris-versicolor centroid: PC1=0.492, PC2=0.549
    Iris-virginica centroid: PC1=1.728, PC2=-0.257
    
    Between-class distances in PCA space:
    Setosa ↔ Versicolor: 2.840
    Setosa ↔ Virginica: 3.948
    Versicolor ↔ Virginica: 1.476
    


```python
# Enhanced Plotly PCA visualization with additional features
fig_pca_enhanced = go.Figure()

# Define colors matching the original matplotlib colors
color_map = {
    'Iris-setosa': '#1f77b4',      # Blue
    'Iris-versicolor': '#ff7f0e',   # Orange  
    'Iris-virginica': '#2ca02c'     # Green
}

# Add scatter plots for each class with enhanced styling
for lab, color in color_map.items():
    class_mask = (y == lab)
    class_data = Y[class_mask]
    
    fig_pca_enhanced.add_trace(go.Scatter(
        x=class_data[:, 0],
        y=class_data[:, 1],
        mode='markers',
        name=lab,
        marker=dict(
            color=color,
            size=10,
            opacity=0.8,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        hovertemplate=
        '<b>%{fullData.name}</b><br>' +
        'PC1: %{x:.3f}<br>' +
        'PC2: %{y:.3f}<br>' +
        'Sample #: %{pointNumber}<br>' +
        '<extra></extra>'
    ))
    
    # Add class centroids
    centroid = np.mean(class_data, axis=0)
    fig_pca_enhanced.add_trace(go.Scatter(
        x=[centroid[0]],
        y=[centroid[1]],
        mode='markers',
        name=f'{lab} Centroid',
        marker=dict(
            color=color,
            size=15,
            symbol='x',
            line=dict(width=3, color='black')
        ),
        showlegend=False,
        hovertemplate=
        '<b>%{fullData.name}</b><br>' +
        'PC1: %{x:.3f}<br>' +
        'PC2: %{y:.3f}<br>' +
        '<extra></extra>'
    ))

# Add variance explanation to the plot
variance_text = f"PC1 explains {var_exp[0]:.1f}% of variance<br>PC2 explains {var_exp[1]:.1f}% of variance<br>Total: {var_exp[0] + var_exp[1]:.1f}%"

fig_pca_enhanced.add_annotation(
    x=0.02, y=0.98,
    xref="paper", yref="paper",
    text=variance_text,
    showarrow=False,
    font=dict(size=12, color="black"),
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="gray",
    borderwidth=1
)

# Update layout with enhanced styling
fig_pca_enhanced.update_layout(
    title={
        'text': 'PCA Visualization: Iris Dataset in Principal Component Space<br><sub>Interactive scatter plot with class centroids</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis=dict(
        title=f'Principal Component 1 ({var_exp[0]:.1f}% variance)',
        # titlefont=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2
    ),
    yaxis=dict(
        title=f'Principal Component 2 ({var_exp[1]:.1f}% variance)',
        # titlefont=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2
    ),
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1
    ),
    plot_bgcolor='white',
    width=900,
    height=700,
    showlegend=True,
    margin=dict(r=150)  # Make room for legend
)

fig_pca_enhanced.show()
```

![PCA](https://github.com/ish-codes-magic/ML-algorithms-from-scratch/blob/main/img/pca_exploratory_5.png)

# PCA class implementation using eigendecomposition


```python
class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X):
        x_std = scaler.fit_transform(X)
        cov_mat = np.cov(x_std.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])
        
        self.components = eig_vecs_sorted[:self.n_components,:]
        
        self.explained_variance_ratio = [i/np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)
        self.matrix_w = np.hstack([eig_pairs[i][1].reshape(4,1) for i in range(self.n_components)])
        return self
    def transform(self, X):
        return X.dot(self.matrix_w)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        return X.dot(self.matrix_w.T)
        
        
```

# PCA class implementation using SVD


```python
class PCA_SVD:
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components = Vt[:self.n_components,:]
        self.explained_variance_ratio = (S**2) / (X.shape[0] - 1)
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)
        return self
    def transform(self, X):
        return X.dot(self.components)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        return X.dot(self.components.T)
        
        
```

# PCA using scikit-learn


```python
from sklearn.decomposition import PCA
X_std = scaler.fit_transform(X)

pca = PCA(n_components = 2).fit(X_std)

print('Components:\n', pca.components_)
print('Explained variance ratio:\n', pca.explained_variance_ratio_)

cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print('Cumulative explained variance:\n', cum_explained_variance)

X_pca = pca.transform(X_std) # Apply dimensionality reduction to X.
print('Transformed data shape:', X_pca.shape)
```

    Components:
     [[ 0.52237162 -0.26335492  0.58125401  0.56561105]
     [ 0.37231836  0.92555649  0.02109478  0.06541577]]
    Explained variance ratio:
     [0.72770452 0.23030523]
    Cumulative explained variance:
     [0.72770452 0.95800975]
    Transformed data shape: (150, 2)
    


```python
# Enhanced Plotly PCA visualization with additional features
fig_pca_enhanced = go.Figure()

# Define colors matching the original matplotlib colors
color_map = {
    'Iris-setosa': '#1f77b4',      # Blue
    'Iris-versicolor': '#ff7f0e',   # Orange  
    'Iris-virginica': '#2ca02c'     # Green
}

# Add scatter plots for each class with enhanced styling
for lab, color in color_map.items():
    class_mask = (y == lab)
    class_data = X_pca[class_mask]
    
    fig_pca_enhanced.add_trace(go.Scatter(
        x=class_data[:, 0],
        y=class_data[:, 1],
        mode='markers',
        name=lab,
        marker=dict(
            color=color,
            size=10,
            opacity=0.8,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        hovertemplate=
        '<b>%{fullData.name}</b><br>' +
        'PC1: %{x:.3f}<br>' +
        'PC2: %{y:.3f}<br>' +
        'Sample #: %{pointNumber}<br>' +
        '<extra></extra>'
    ))
    
    # Add class centroids
    centroid = np.mean(class_data, axis=0)
    fig_pca_enhanced.add_trace(go.Scatter(
        x=[centroid[0]],
        y=[centroid[1]],
        mode='markers',
        name=f'{lab} Centroid',
        marker=dict(
            color=color,
            size=15,
            symbol='x',
            line=dict(width=3, color='black')
        ),
        showlegend=False,
        hovertemplate=
        '<b>%{fullData.name}</b><br>' +
        'PC1: %{x:.3f}<br>' +
        'PC2: %{y:.3f}<br>' +
        '<extra></extra>'
    ))

# Add variance explanation to the plot
variance_text = f"PC1 explains {var_exp[0]:.1f}% of variance<br>PC2 explains {var_exp[1]:.1f}% of variance<br>Total: {var_exp[0] + var_exp[1]:.1f}%"

fig_pca_enhanced.add_annotation(
    x=0.02, y=0.98,
    xref="paper", yref="paper",
    text=variance_text,
    showarrow=False,
    font=dict(size=12, color="black"),
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="gray",
    borderwidth=1
)

# Update layout with enhanced styling
fig_pca_enhanced.update_layout(
    title={
        'text': 'PCA Visualization: Iris Dataset in Principal Component Space<br><sub>Interactive scatter plot with class centroids</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis=dict(
        title=f'Principal Component 1 ({var_exp[0]:.1f}% variance)',
        # titlefont=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2
    ),
    yaxis=dict(
        title=f'Principal Component 2 ({var_exp[1]:.1f}% variance)',
        # titlefont=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2
    ),
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1
    ),
    plot_bgcolor='white',
    width=900,
    height=700,
    showlegend=True,
    margin=dict(r=150)  # Make room for legend
)

fig_pca_enhanced.show()
```

![PCA](https://github.com/ish-codes-magic/ML-algorithms-from-scratch/blob/main/img/pca_exploratory_6.png)


Note for SVD:

|Step	        | SVD Role	                | PCA Interpretation                   |
|-------------- | ---------                 |-----------                           |
|Center Data	| Preprocessing	            | Removes mean                         |
|Decompose (X)	| SVD X=UΣVT                | Finds directions of maximal variance |
|V              | Right singular vectors	| Principal component axes             |
|Σ2             | Singular values squared	| Variance explained (eigenvalues)     |
|Project Data	| UΣ or XV	                | Principal component scores           |


```python

```
