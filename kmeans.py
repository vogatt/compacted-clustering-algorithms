from math import sqrt, pi, sin
import random
import itertools


def euclidean_distance(a, b):
	""" 
	Calculates an euclidean distance beetween two data points. 

	"""
	if(not isinstance(a, list) or not isinstance(b, list)):
		raise TypeError("Input vectors should be list objects")
	if(len(a) != len(b)):
		raise Exception("Input vectors must have the same length")
	return sqrt(sum([pow(x-y,2) for x,y in itertools.izip(a,b)]))


def rss(input_vectors, centroids):
	""" 
	Calculates a residual sum of squares over all cluster centroids 
	and all vectors.
	
	"""
	return sum( [pow(euclidean_distance(x['data'], centroids[x['centroid']]),2) \
			for x in input_vectors] )


def kmeans_clustering(k, input_vectors, max_iterations, epsilon_rss):
	""" 
	
	Performs a k-means clustering on the input_vectors stopping after
	max_iterations or after reaching delta_rss that is less than epsilon_rss. 
	
	"""
	centroids = random.sample(input_vectors, k)	
	input_vectors = map( \
			lambda w: {\
				'data': w,\
				'centroid': min( \
					[(euclidean_distance(w, x), c) for c,x in enumerate(centroids)])[1]}, \
			input_vectors)
	current_rss = rss(input_vectors, centroids)
	
	for iteration in xrange(max_iterations):		
		for index,cent in enumerate(centroids):
		
			local_vec = filter(lambda x: x['centroid'] == index, input_vectors)
			centroids[index] = [x/len(local_vec) \
					for x in reduce(\
						lambda x,y: \
							{'data':[a + b for a,b in itertools.izip(x['data'],y['data'])]},\
						local_vec )['data'] ]
		
		for index, vec in enumerate(input_vectors):
		
			input_vectors[index]['centroid'] = min( \
					[(euclidean_distance(vec['data'], x), c) \
					for c,x in itertools.izip(range(k),centroids)])[1]
		
		previous_rss = current_rss	
		current_rss = rss(input_vectors, centroids)
		if((previous_rss - current_rss) < epsilon_rss):	break

	return input_vectors

def make_random_vectors(n,k,max_value):
	"""
	Generates a list of n k-dimensional vectors, 
	whose coordinets are within a [-max_value, max_value] range.
	
	"""
	vectors=[]
	for a in xrange(n):
		vec = []
		for w in xrange(k):
			vec.append(sin(random.random() * 2.0 * pi) * max_value)
		vectors.append(vec)

	return vectors


