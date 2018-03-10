# Tudor Berariu, 2016

from sys import argv
from zipfile import ZipFile
from random import randint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers
from mpl_toolkits.mplot3d import Axes3D

def getArchive():
    archive_url = "http://www.uni-marburg.de/fb12/datenbionik/downloads/FCPS"
    local_archive = "FCPS.zip"
    from os import path
    if not path.isfile(local_archive):
        import urllib.request
        print("downloading...")
        urllib.request.urlretrieve(archive_url, filename=local_archive)
        assert(path.isfile(local_archive))
        print("got the archive")
    return ZipFile(local_archive)

def getDataSet(archive, dataSetName):
    path = "FCPS/01FCPSdata/" + dataSetName

    lrnFile = path + ".lrn"
    with archive.open(lrnFile, "r") as f:                       # open .lrn file
        N = int(f.readline().decode("UTF-8").split()[1])    # number of examples
        D = int(f.readline().decode("UTF-8").split()[1]) - 1 # number of columns
        f.readline()                                     # skip the useless line
        f.readline()                                       # skip columns' names
        Xs = np.zeros([N, D])
        for i in range(N):
            data = f.readline().decode("UTF-8").strip().split("\t")
            assert(len(data) == (D+1))                              # check line
            assert(int(data[0]) == (i + 1))
            Xs[i] = np.array(list(map(float, data[1:])))

    clsFile = path + ".cls"
    with archive.open(clsFile, "r") as f:                        # open.cls file
        labels = np.zeros(N).astype("uint")

        line = f.readline().decode("UTF-8")
        while line.startswith("%"):                                # skip header
            line = f.readline().decode("UTF-8")

        i = 0
        while line and i < N:
            data = line.strip().split("\t")
            assert(len(data) == 2)
            assert(int(data[0]) == (i + 1))
            labels[i] = int(data[1])
            line = f.readline().decode("UTF-8")
            i = i + 1

        assert(i == N)

    return Xs, labels                          # return data and correct classes

def get_closest_neighbour(centroids, x):
	all_dists = np.sum((centroids - x)**2, axis=1)
	return np.argmin(all_dists)

def choose_random_centroids(K, Xs):
	################ TODO 1.1 ########################

	# choose K random elements from Xs to be centroids
    idx = np.arange(Xs.shape[0])
    np.random.shuffle(idx)
    centroids = Xs[idx[:K]]

    ####################################################

    return centroids

def kMeansPlusPlus(K, Xs):
	################### TODO 3 #########################

	(N, D) = Xs.shape
	centroid_idx = np.random.randint(N)
	centroids = np.array([Xs[centroid_idx]])
	used_indices = np.array([centroid_idx])
	cnt = 1
	while cnt < K:
		# function to compute the minimum distance to an existent centroid
		func = lambda x : np.amin(np.sum((centroids - x)**2, axis=1))

		# compute distance from each data point to its closest centroid
		clusters_dist = np.fromiter((func(x) for x in Xs), "uint", N)

		# distances to centroids that have already been used shouldn't count
		clusters_dist[used_indices] = 0

		# the next centroid is chosen from a weighted probability distribution
		# where a data point is chosen with probability proportional to the 
		# distance to its closest centroid
		probs = clusters_dist**2 / np.sum(clusters_dist**2)
		
		# compute the cumulative probabilities, choose a random number 
		# between 0 and 1 and find the first index where the cumulative
		# probability is greater -> this is the index of the new centroid
		cumul_probs = np.cumsum(probs)
		rand_num = np.random.random()
		rand_idx = np.argwhere(cumul_probs >= rand_num)[0]

		# mark the current index
		used_indices = np.append(used_indices, rand_idx)

		# store the new centroid
		centroids = np.append(centroids, Xs[rand_idx])
		centroids = centroids.reshape(-1, D)
		cnt += 1
	
	return centroids

	####################################################

def kMeans(K, Xs, random_centroids=True):
    (N, D) = Xs.shape
    if K > N:
    	print("Cannot sample K different centroids. Exiting")
    	sys.exit(1)

    #################### TODO 1 #######################

    if random_centroids:
    	centroids = choose_random_centroids(K, Xs)
    else:
    	centroids = kMeansPlusPlus(K, Xs)

    # closest neighbor function
    func = lambda x : get_closest_neighbour(centroids, x)	
    clusters = np.fromiter((func(x) for x in Xs), "uint", N)

    prev_clusters = np.copy(clusters)
    converged = False

    count_iterations = 0
    while not converged:
    	unique, counts = np.unique(clusters, return_counts=True)
    	cluster_counts = dict(zip(unique, counts))

    	# compute the new centroids: 
    	# new centroid = average of all its current elements
    	for k in range(K):
    		cluster_sum = np.sum(Xs[clusters == k], axis=0)
    		new_centroid = cluster_sum / cluster_counts[k]
    		centroids[k] = new_centroid
    	
    	# compute new cluster idx for each element from the dataset
    	func = lambda x : get_closest_neighbour(centroids, x)	
    	clusters = np.fromiter((func(x) for x in Xs), "uint", N)
    	
    	# convergence test
    	diff = np.sum(clusters - prev_clusters)
    	if diff == 0:
    		converged = True
    	prev_clusters = np.copy(clusters)

    	count_iterations += 1

    print("Clusters converged after ", count_iterations, "iterations")

    ###################################################
    
    return clusters, centroids

def randIndex(clusters, labels):

	################## TODO 2 ########################
    tp, fp, tn, fn = 0, 0, 0, 0
    N = clusters.shape[0]
    for i in range(N):
    	for j in range(N):
    		if i == j: 
    			continue
    		if labels[i] == labels[j]:
    			if clusters[i] == clusters[j]:
    				tp += 1
    			else:
    				fn += 1
    		else:
    			if clusters[i] == clusters[j]:
    				fp += 1
    			else:
    				tn += 1

    r = 1.0 * (tp + tn) / (tp + tn + fp + fn)
    #################################################

    return r

def plot(Xs, labels, K, clusters):
    labelsNo = np.max(labels)
    markers = []                                     # get the different markers
    while len(markers) < labelsNo:
        markers.extend(list(matplotlib.markers.MarkerStyle.filled_markers))
    colors = plt.cm.rainbow(np.linspace(0, 1, K+1))

    if Xs.shape[1] == 2:
        x = Xs[:,0]
        y = Xs[:,1]
        for (_x, _y, _c, _l) in zip(x, y, clusters, labels):
            plt.scatter(_x, _y, s=500, c=colors[_c], marker=markers[_l])
        plt.scatter(centroids[:,0], centroids[:, 1],
                    s=800, c=colors[K], marker=markers[labelsNo]
        )
        plt.show()

    elif Xs.shape[1] == 3:
        x = Xs[:,0]
        y = Xs[:,1]
        z = Xs[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=colors[_c], marker=markers[_l])
        ax.scatter(centroids[:,0], centroids[:, 1], centroids[:, 2],
                    s=400, c=colors[K], marker=markers[labelsNo]
        )
        plt.show()

    else:
        for i in range(N1):
            print(i, ": ", clusters[i], " ~ ", labels[i])

if __name__ == "__main__":
    if len(argv) < 3:
        print("Usage: " + argv[0] + " dataset_name K")
        exit()

    Xs, labels = getDataSet(getArchive(), argv[1])    # Xs is NxD, labels is Nx1
    K = int(argv[2])                                # K is the numbe of clusters

    clusters, centroids = kMeans(K, Xs)
    print("k-Means randIndex: ", randIndex(clusters, labels))
    plot(Xs, labels, K, clusters)

    clusters, centroids = kMeans(K, Xs, random_centroids=False)
    print("k-Means++ randIndex: ", randIndex(clusters, labels))
    plot(Xs, labels, K, clusters)
