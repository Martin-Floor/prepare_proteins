import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from matplotlib import pyplot as plt

class cluster:
    """
    Class to hold methods for generating clusters based on a distance matrix given as input.

    The method uses the cluster hierarchical package from scipy and methods within.

    Attributes
    ----------
    distances : numpy.ndarray
        Array containing the set of pairwise distances to use for clustering
    verbose : bool
        Whether to print the outputs of cluster calculations.
    linkage : numpy.ndarray
        Hierarchical clustering encoded as a linkage matrix.
    clusters : dict
        Dictionary containing the elements of each cluster or its centroids.

    Methods
    -------
    getClusters()
        Get clusters based on a threshold distances
    plotDendrogram()
        Plot dendogram containing the clustering information
    """

    def __init__(self, distance_matrix, verbose=False):
        """
        At initialization the method calls the linkage method to generate an initialization
        set of clusters.

        Parameters
        ----------
        distance_matrix : numpy.ndarray
            Array containing the set of pairwise distances to use for clustering
        verbose : bool (False)
            Whether to print clustering information.
        """

        # Define attributes
        self.distances = distance_matrix
        self.verbose = verbose
        # Define attributes holders
        self.clusters = None
        self.ordered_labels = None

        squareform_distances = squareform(distance_matrix, checks=False)

        # Calculate clusters
        self.linkage = linkage(squareform_distances , method='average')

        if verbose:
            # Ignore diagonal values for max and min printing
            mask = np.ones(self.distances.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            max_value = self.distances[mask].max()
            min_value = self.distances[mask].min()

            print('Max pairwise distance: %.3f [diagonal values ignored]' % max_value)
            print('Min pairwise distance: %.3f [diagonal values ignored]' % min_value)

    def getClustersByDistance(self, clustering_distance, return_centroids=False):
        """
        Define clusters based on a threshold distances

        Parameters
        ----------
        clustering_distance : float
            Threshold  value to cluster matrix distance elements.
        return_centroids : bool (False)
            Return only the centroids of each cluster.

        Returns
        -------
        clusters : dict
            Dictionary containing the elements of each cluster or only the centroids
            if return_centroids parameter is set to true.
        """

        self.clusters = fcluster(self.linkage, clustering_distance, criterion='distance')

        if self.verbose:
            print('There are '+str(len(set(self.clusters)))+' clusters at clustering distance '+str(clustering_distance))

        C = { x+1:[] for x in range(len(set(self.clusters))) }
        for i,c in enumerate(self.clusters):
            C[c].append(i)

        if return_centroids == True:
            for c in C:
                if len((C[c])) > 1:
                    added_distances = []
                    for i in C[c]:
                        added_distances.append((i,np.sum(self.distances[i])))
                    C[c] = sorted(added_distances, key=lambda x:x[1])[0][0]
                else:
                    C[c] = C[c][0]

        self.clusters = C

        return C

    def getNClusters(self, number_of_clusters, return_centroids=False):
        """
        Define clusters based on a threshold distances

        Parameters
        ----------
        number_of_clusters : float
            Number of clusters to output.
        return_centroids : bool (False)
            Return only the centroids of each cluster.

        Returns
        -------
        clusters : dict
            Dictionary containing the elements of each cluster or only the centroids
            if return_centroids parameter is set to true.
        """

        self.clusters = fcluster(self.linkage, number_of_clusters, criterion='maxclust')

        C = { x+1:[] for x in range(len(set(self.clusters))) }
        for i,c in enumerate(self.clusters):
            C[c].append(i)

        if return_centroids == True:
            for c in C:
                if len((C[c])) > 1:
                    added_distances = []
                    for i in C[c]:
                        added_distances.append((i,np.sum(self.distances[i])))
                    C[c] = sorted(added_distances, key=lambda x:x[1])[0][0]
                else:
                    C[c] = C[c][0]

        self.clusters = C

        return C

    def plotDendrogram(self, labels=None, dpi=100, figsize=None):
        """
        Plot dendogram containing the clustering information

        Parameters
        ----------
        dpi : int
            Resolution of the generated plot.
        """

        plt.figure(dpi=dpi, figsize=figsize)
        dendrogram(self.linkage)

        if labels != None:
            self.ordered_labels = [labels[int(x.get_text())] for x in plt.xticks()[1]]
            positions = plt.xticks()[0]
            l = plt.xticks(positions, self.ordered_labels)
