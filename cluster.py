import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from dtaidistance import dtw
from empresa4.datasets import get_dataset
from sklearn.metrics import pairwise_distances_argmin_min

def find_optimal_clusters(dist_matrix, min_clusters=5, max_clusters=15, save_path='~/buckets/b1/datasets/kmeans.png'):
    inertia = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        print(f"Running KMeans with {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(dist_matrix)
        inertia.append(kmeans.inertia_)

    plt.plot(range(min_clusters, max_clusters + 1), inertia)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    
    # Save the plot before showing it
    plt.savefig(save_path)

    plt.show()

    # Here you manually pick the optimal number of clusters from the plot
    optimal_k = int(input("Enter the optimal number of clusters (based on the plot): "))
    return optimal_k

def main():
    # Step 1: Open the distances file and the time series dataset
    distances_df = pd.read_csv('~/buckets/b1/datasets/distances.csv')
    time_series_df = get_dataset('time_series_dataset.csv', index_col=[0,1])

    # Step 2: Identify computed and uncomputed time series
    computed_indices = distances_df.columns
    all_indices = time_series_df.index
    uncomputed_indices = all_indices.difference(computed_indices)
    print(f"Number of computed time series: {len(computed_indices)}")
    print(f"Number of uncomputed time series: {len(uncomputed_indices)}")
    print(f"Total number of time series: {len(all_indices)} == {len(computed_indices) + len(uncomputed_indices)}")

    # Step 3: Find optimal number of clusters
    optimal_k = find_optimal_clusters(distances_df.values)

    # Step 4: Clusterize using the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k)
    clusters = kmeans.fit_predict(distances_df.values)

    # Step 5: Find the time series closest to the center of each cluster
    centers = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centers, distances_df.values)
    closest_series = distances_df.columns[closest]

    # Step 6: Assign uncomputed time series to closest cluster
    cluster_assignments = {index: cluster for index, cluster in zip(computed_indices, clusters)}
    for idx in uncomputed_indices:
        min_dist = float('inf')
        closest_cluster = None
        for center_idx in closest_series:
            dist = dtw.distance_fast(time_series_df.loc[idx].values, time_series_df.loc[center_idx].values)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_assignments[center_idx]
        cluster_assignments[idx] = closest_cluster

    # Step 7: Save the final cluster assignments to a CSV file
    cluster_assignments_df = pd.DataFrame.from_dict(cluster_assignments, orient='index', columns=['Cluster'])
    cluster_assignments_df.to_csv('~/buckets/b1/datasets/clusterized_time_series.csv')

if __name__ == '__main__':
    main()
