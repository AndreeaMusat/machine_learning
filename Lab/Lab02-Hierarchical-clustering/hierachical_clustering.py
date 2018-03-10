# Tudor Berariu, 2016

from sys import argv
from sys import exit
from zipfile import ZipFile
from random import randint

import numpy as np
from scipy.cluster import hierarchy
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

#################################################################
####################### TODO 1 + 2 ##############################
def hierarchicalClustering(Xs, method, metric='euclidean'):
	(N, D) = Xs.shape
	Z = np.zeros((N-1, 4))
	INF = 1e20

	# clusters[i] = index of cluster in which data point i is
	clusters = np.array([i for i in range(N)])
	
	if metric=='euclidean':
		all_dists = np.sqrt(np.sum((Xs[:,np.newaxis] - Xs)**2, axis=-1))
	elif metric=='cityblock':
		all_dists = np.sum(np.abs(Xs[:,np.newaxis] - Xs), axis=-1)
	else:
		print("Metric unknown. Exiting...")
		exit(1)

	all_dists[np.where(all_dists==0)] = INF

	cnt = 0
	while cnt < N - 1:

		# always get the most similar clusters (find min elem in all_dists)
		idx1, idx2 = np.unravel_index(np.argmin(all_dists, axis=None), all_dists.shape)
		min_dist = np.amin(all_dists, axis=None)

		# update the cluster index for each element in cluster idx1 or idx2
		clusters[np.logical_or(clusters==idx1, clusters==idx2)] = N + cnt

		# create a new entry in Z
		num_elems = len(clusters[np.logical_or(clusters==idx1, clusters==idx2)])
		Z[cnt] = np.array([idx1, idx2, min_dist, num_elems])
			
		# create a new distance matrix (add a new row and column for the new cluster)
		new_all_dists = np.zeros((all_dists.shape[0]+1, all_dists.shape[1]+1))
		new_all_dists[:-1,:-1] = all_dists
		
		# now find the new distances from evary cluster to the newly formed one

		# single linkage metric: new distance from every cluster to the new cluster 
		# is the minimum of the distances to the merged clusters
		if method == 'single':		
			new_dists = np.amin(all_dists[np.array([idx1, idx2]), :], axis=0)

		# complete linkage: new_dist = max(cluster_dist)
		elif method == 'complete':	
			new_dists = np.amax(all_dists[np.array([idx1, idx2]), :], axis=0)

		# group average linkage
		elif method == "average":		
			n_idx1 = len(clusters==idx1)
			n_idx2 = len(clusters==idx2)
			new_dists = n_idx1 * all_dists[idx1, :] + n_idx2 * all_dists[idx2, :]
			new_dists /= (n_idx1 + n_idx2)
		else:
			print("Method %s unkown. Exiting" % method)
			exit(1)

		# update the new distances 
		new_all_dists[-1,:-1] = new_dists
		new_all_dists[:-1,-1] = new_dists

		# make sure to set distances from clusters idx1 and idx2 to INF
		# (they do not exist anymore)
		new_all_dists[-1,-1] = INF
		new_all_dists[np.array([idx1, idx2]), :] = INF
		new_all_dists[:, np.array([idx1, idx2])] = INF
		all_dists = new_all_dists

		cnt += 1

	return Z

#####################################################################
#####################################################################


#####################################################################
########################## TODO 4 ###################################
def extractClusters(Xs, Z):
    (N, D) = Xs.shape
    assert(Z.shape == (N-1, 4))

    # get the index of the maximum-distance merge
    max_merge_dist = np.amax(Z[:,2])
    threshold = max_merge_dist - 0.8 * np.average(Z[:,2])
    max_merge_idx = np.argwhere(Z[:,2]>=threshold)[0][0]

    # compute the number of clusters
    K = N - max_merge_idx

    # simulate clusters evolution given Z
    clusters = np.array([i for i in range(N)])    
    for i in range(max_merge_idx):
    	clusters[np.logical_or(clusters==Z[i,0], clusters==Z[i, 1])] = N+i

    # cluster ids should be in range 0..K-1
    unique = np.unique(clusters)
    new_clusters = np.zeros(clusters.shape)
    for i in range(len(unique)):
    	new_clusters[clusters==unique[i]] = i

    return K, new_clusters.astype(int)

####################################################################
####################################################################

def randIndex(clusters, labels):
    assert(labels.size == clusters.size)
    N = clusters.size

    a = 0.0
    b = 0.0

    for (i, j) in [(i,j) for i in range(N) for j in range(i+1, N) if i < j]:
        if ((clusters[i] == clusters[j]) and (labels[i] == labels[j]) or
            (clusters[i] != clusters[j]) and (labels[i] != labels[j])):
            a = a + 1
        b = b + 1

    return float(a) / float(b)

def plot(Xs, labels, K, clusters):
    labelsNo = np.max(labels)
    markers = []                                     # get the different markers
    while len(markers) < labelsNo:
        markers.extend(list(matplotlib.markers.MarkerStyle.filled_markers))
    colors = plt.cm.rainbow(np.linspace(0, 1, K+1))

    fig = plt.figure()
    if Xs.shape[1] == 2:
        x = Xs[:,0]
        y = Xs[:,1]
        for (_x, _y, _c, _l) in zip(x, y, clusters, labels):
            plt.scatter(_x, _y, s=200, c=colors[_c], marker=markers[_l])
        plt.show()
    elif Xs.shape[1] == 3:
        x = Xs[:,0]
        y = Xs[:,1]
        z = Xs[:,2]
        ax = fig.add_subplot(111, projection='3d')
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=colors[_c], marker=markers[_l])
        plt.show()
    else:
        for i in range(N1):
            print(i, ": ", clusters[i], " ~ ", labels[i])


if __name__ == "__main__":
	if len(argv) < 2:
		print("Usage: " + argv[0] + " dataset_name")
		exit()

	Xs, labels = getDataSet(getArchive(), argv[1])    # Xs is NxD, labels is Nx1
    
    ################# check all the methods #########################
	for method in ['single', 'complete', 'average']:
		Z = hierarchicalClustering(Xs, method=method, metric='euclidean') 
		
		if not hierarchy.is_valid_linkage(Z):
			print("Error. Your linkage is not valid. Exiting...")
			exit(1)

		plt.figure()
		plt.title('My dendrogram ' + method)
		dn = hierarchy.dendrogram(Z)

		plt.figure()
		plt.title('SciPy dendrogram ' + method)
		dn = hierarchy.dendrogram(hierarchy.linkage(Xs, method=method, metric='euclidean'))

		K, clusters = extractClusters(Xs, Z)
		print("randIndex: ", randIndex(clusters, labels))
		plot(Xs, labels, K, clusters)
    #################################################################
