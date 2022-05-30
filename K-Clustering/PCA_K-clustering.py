import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA
import pandas as pd

# Each column units are kg in a m^3 mixture
# Except the Age column, which is in days
# and the Compressive stregth column, which is in Megapascals

# Read data as a dataframe with Pandas
data = pd.read_excel(r"C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\10mo Semestre\SI\FinalProject\Datasets\Concrete\Concrete_Data.xls")

# Transform Pandas dataframe to numpy array
np_data = data.to_numpy()

# Define X and y for PCA algorithm
X = np_data
# y = np_data[:, 8]

# Print shapes
print("X shape is:", X.shape)
#print("y shape is:", y.shape)

# Reduce dimensionality
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X:', X.shape)
print('Shape of transformed X:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# Expand dimensions of principal component vectors to concatenate them and
# pass as data to the KMeans algorithm
x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2,axis=1)

X = (np.append(x1, x2, axis=1))

# set random seed for consistent results across testing
np.random.seed(45)

# import KMeans class
from KMeansClustering import KMeans, euclidean_distance

# print(type(X))
# print(X.shape)
# print(X[0])


k = KMeans(K=3, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
print(y_pred[:10])



# Print indices of X in each cluster
# for _ in range(k.K):
#     print("\n", k.clusters[_], "\n")

# User Queries
features = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age (days)", "Compressive Strength"]
x = np.zeros((9,1))
Predict = input("Do you want to make a prediction? (y/n)")
while Predict == 'y':
    for i in range(len(x)):
        if i < 7:
            x[i] = float(input(f"Enter kg of {features[i]} per m^3: "))
        if i == 7:
            x[i] = float(input(f"Enter {features[i]}: "))
        elif i == 8:
            x[i] = float(input(f"Enter {features[i]}: "))
            break

    # x = [540, 0, 0, 162, 2.5, 1040, 676, 28, 79.99]       #First example
    # x = [198.6, 132.4, 0, 192, 0, 978.4, 825.5, 360, 44.3]      #Second example
    x = np.squeeze(np.array(x))
    # print(x)
    x_projected = pca.transform(x)

    x1 = x_projected[0]
    x2 = x_projected[1]

    # print(x_projected[:10])
    # print(X_projected[:10])

    # Make predictions by computing the closest point to the user entered data
    # and assigning that user data point to the corresponding cluster
    distances = []
    for X_instance in X_projected:
        distance = euclidean_distance(x_projected, X_instance)
        distances.append(distance)

    #print(np.expand_dims(np.array(distances[:10]), axis=1))
    X_index = np.argmin(distances)
    print("\nX_index: ", X_index)
    #print(X_index)

    for cluster_idx in range(k.K):
        if X_index in k.clusters[cluster_idx]:
            pred_cluster = cluster_idx

    print("Predicted cluster: ", pred_cluster)

    Predict = input("Make another prediction? (y/n)")
