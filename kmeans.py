from sklearn import datasets
import numpy as np
from numpy import random
from matplotlib import pyplot as plt


def random_indices(k, bound):
	# Creates k distinct, random integers between 0 and bound.
	# n = number of indices, bound = largest index
	index_set = set()
	while len(index_set) < k:
		index_set.add(np.random.randint(bound))
	return np.array(list(index_set))

def cluster_assignment(means, data):
	# Given k means, we create a clustering about those means.
	# Returns a cluster assignment index and the loss (aka energy)
	# of the cluster. 
	k = len(means)
	data_size = data.shape[0]
	data_dimension = data.shape[1]
	tiled_means = np.tile(means, data_size).reshape(k, data_size, data_dimension)
	# ith column of sq_distances is distances from all points to ith mean
	# the ith element of cluster_index 
	# is the cluster to which the ith data point belongs
	sq_distances = np.sqrt(np.sum((tiled_means - data)**2, axis=2)).T
	loss = np.sum(sq_distances[:,0])
	cluster_index = sq_distances.argsort(axis=1)[:, 0]
	return cluster_index, loss

def updated_means(cluster_index, data):
	# subfunction of clustering(k, data)
	# Given a current cluster assignment,
	# returns the (possibly new) means of each cluster.
	k = len(np.unique(cluster_index))
	new_means = np.reshape([], (0, data.shape[1]))
	for i in range(k):
		#if i not in cluster_index:
		#	print "One of the means was too far away."
		#	break
		single_cluster = data[np.where(cluster_index == i)]
		cluster_mean = np.average(single_cluster, axis=0)
		new_means = np.append(new_means, [cluster_mean], axis=0)
	return  new_means

def sets_not_equal(A,B):
	A = set(map(tuple, A))
	B = set(map(tuple,B))
	return A != B

def clustering(k, data):
	# Given k initial means, we recalculate new means
	# until stability is reached, yielding a clustering (i.e., partition).  
	# Returns the cluster means, clustering assingment, and loss (or energy)
	# of the cluster.
	old_means = data[random_indices(k, data.shape[0])]
	means_to_update = old_means
	first_time = True
	# update the means until they are stable
	while first_time or sets_not_equal(old_means, means_to_update):
		cluster_index = (cluster_assignment(means_to_update, data))[0]
		old_means = means_to_update
		means_to_update = updated_means(cluster_index, data)
		first_time = False
	final_means = means_to_update
	final_cluster_index, loss = cluster_assignment(final_means, data)
	return final_means, final_cluster_index, loss

def partition(cluster_index, data):
	k = len(np.unique(cluster_index))
	P = []
	for i in range(k):
		P.append(data[np.where(cluster_index == i)])
	return P



def create_clustering(k, data, m):
	# Creates m clusterings of the data,
	# then returns the cluster assignment index
	# of the clustering with smallest loss.
	means_array = []
	cluster_array = []
	loss_array = []
	for i in range(m):
		means, index, loss = clustering(k, data)
		means_array.append(means)
		cluster_array.append(index)
		loss_array.append(loss)
	min_index = np.array(loss_array).argsort()[0]
	cluster_index = cluster_array[min_index]
	means = means_array[min_index]
	return cluster_index, means #, partition(k, cluster_index, data)


def plot_iris(C):
	# C is an cluster index
	D = iris.data
	fig = plt.figure()
	subplot1 = fig.add_subplot(2,2,1)
	plt.scatter( D[np.where(iris.target == 0)][:, 0], D[np.where(iris.target == 0)][:, 1], color='k', label=iris.target_names[0])
	plt.scatter( D[np.where(iris.target == 1)][:, 0], D[np.where(iris.target == 1)][:, 1], color='r', label=iris.target_names[1])
	plt.scatter( D[np.where(iris.target == 2)][:, 0], D[np.where(iris.target == 2)][:, 1],  color='b', label=iris.target_names[2])
	subplot1.set_xlabel(iris.feature_names[0])
	subplot1.set_ylabel(iris.feature_names[1])
	subplot1.set_xticks([4, 6, 8])
	subplot1.set_yticks([2, 4])
	subplot1.set_title("Actual Species Clustering")
	subplot1.legend(loc='best')


	subplot2 = fig.add_subplot(2,2,2)
	plt.scatter( D[np.where(iris.target == 0)][:, 2], D[np.where(iris.target == 0)][:, 3], color='k', label = iris.target_names[0])
	plt.scatter( D[np.where(iris.target == 1)][:, 2], D[np.where(iris.target == 1)][:, 3], color='r', label = iris.target_names[1])
	plt.scatter( D[np.where(iris.target == 2)][:, 2], D[np.where(iris.target == 2)][:, 3], color='b', label = iris.target_names[2])
	subplot2.set_xlabel(iris.feature_names[2])
	subplot2.set_ylabel(iris.feature_names[3])
	subplot2.set_title("Actual Species Clustering")
	subplot2.set_xticks([0, 4, 8])
	subplot2.set_yticks([0, 2])
	

	subplot3 = fig.add_subplot(2,2,3)
	plt.scatter( D[np.where(C == 0)][:, 0], D[np.where(C == 0)][:, 1] , color='Chocolate')
	plt.scatter( D[np.where(C == 1)][:, 0], D[np.where(C == 1)][:, 1] , color='DarkSlateBlue')
	plt.scatter( D[np.where(C == 2)][:, 0], D[np.where(C == 2)][:, 1] , color='ForestGreen')
	subplot3.set_xlabel(iris.feature_names[0])
	subplot3.set_ylabel(iris.feature_names[1])
	subplot3.set_xticks([4,  6, 8])
	subplot3.set_yticks([2, 4])
	subplot3.set_title("k-means Clustering")
	

	subplot4 = fig.add_subplot(2,2,4)
	plt.scatter( D[np.where(C == 0)][:, 2], D[np.where(C == 0)][:, 3] , color='Chocolate')
	plt.scatter( D[np.where(C == 1)][:, 2], D[np.where(C == 1)][:, 3] , color='DarkSlateBlue')
	plt.scatter( D[np.where(C == 2)][:, 2], D[np.where(C == 2)][:, 3] , color='ForestGreen')
	subplot4.set_xlabel(iris.feature_names[2])
	subplot4.set_ylabel(iris.feature_names[3])
	subplot4.set_title("k-means Clustering")
	subplot4.set_xticks([0, 4, 8])
	subplot4.set_yticks([0, 2])
    
    

def plot_iris2(i,j, C):
	# Takes two indices and plots the clustering
	# against those axes.
	D = iris.data
	fig = plt.figure()
	subplot = fig.add_subplot(1,1,1)
	plt.scatter( D[np.where(C == 0)][:, i], D[np.where(C == 0)][:, j] , color='c')
	plt.scatter( D[np.where(C == 1)][:, i], D[np.where(C == 1)][:, j] , color='g')
	plt.scatter( D[np.where(C == 2)][:, i], D[np.where(C == 2)][:, j] , color='y')
	subplot.set_xlabel(iris.feature_names[i])
	subplot.set_ylabel(iris.feature_names[j])
	subplot.set_title("Clustering")
	subplot.set_xticks([])
	subplot.set_yticks([])

def main():
    cluster_index, means = create_clustering(3, iris.data, 1000)
    print cluster_index
    plot_iris(cluster_index)
    plt.savefig("iris_clustering.png")
    plt.show()
    
    

if __name__ == "__main__":
    iris = datasets.load_iris()
    main()
    

	
