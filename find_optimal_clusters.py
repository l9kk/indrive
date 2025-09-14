"""Script to find optimal number of clusters using elbow method."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def find_optimal_clusters():
    """Find optimal number of clusters using elbow method and silhouette analysis."""
    
    # Load data
    data_path = "data/geo_locations_labeled_advanced.csv"
    df = pd.read_csv(data_path)
    
    # Extract destination points (type B)
    B_points = df[df['point_type'] == 'B'][['lat', 'lng']].copy()
    print(f"Analyzing {len(B_points)} destination points...")
    
    # Test different cluster numbers
    k_range = range(2, 21)  # Test from 2 to 20 clusters
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"Testing {k} clusters...")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(B_points)
        
        # Calculate inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(B_points, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"  Inertia: {kmeans.inertia_:.2f}, Silhouette: {silhouette_avg:.3f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette plot
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k
    # Method 1: Highest silhouette score
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    # Method 2: Elbow detection (simplified)
    # Calculate second derivative to find elbow
    second_derivatives = np.diff(inertias, 2)
    elbow_k = k_range[np.argmax(second_derivatives) + 2]
    
    print(f"\n🎯 RESULTS:")
    print(f"Best k by Silhouette Score: {best_k_silhouette} (score: {max(silhouette_scores):.3f})")
    print(f"Best k by Elbow Method: {elbow_k}")
    print(f"Current k in config: 7")
    
    # Detailed analysis for different k values
    print(f"\n📊 DETAILED ANALYSIS:")
    for i, k in enumerate(k_range):
        print(f"k={k:2d}: Inertia={inertias[i]:8.0f}, Silhouette={silhouette_scores[i]:.3f}")
    
    return best_k_silhouette, elbow_k

if __name__ == "__main__":
    find_optimal_clusters()