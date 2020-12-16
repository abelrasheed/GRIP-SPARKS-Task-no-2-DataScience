import math
import random
import numpy

def distance(x1 , x2):
    sum = 0
    for i,j in zip(x1,x2):
        temp = math.pow((i-j),2)
        sum += temp 
    sum = math.sqrt(sum)
    return sum

class Kmeans_own:
    def __init__(self,X,K=3,num_iterations=500):
        self.K = K
        self.X = X
        self.m, self.n = self.X.shape
        self.clusters = [0]*self.m
        self.num_iterations = num_iterations
        self.centroids = []

    def train(self):
        for k in range(self.K):
            temp = initialize_centroids(self.n)
            self.centroids.append(temp)
        for i in range(self.num_iterations):
            # self.clusters = []
            for j  in range(self.m): #traversing through the examples
                for k in range(self.K): #traversing through the cetroids for each example
                    if(k == 0):
                        short_dist = distance(self.centroids[k],self.X[j])
                        self.clusters[j] = k
                    else:
                        dist = distance(self.centroids[k],self.X[j])                       
                        if(dist<short_dist):                            
                            short_dist = dist
                            self.clusters[j] = k
            # updating cetroids
            # for j in range(self.m):
            #     for k in range(self.K):
            #         indices = []
            #         for ind in range(self.m):
            #             if(self.clusters[ind]==k):
            #                 indices.append(ind)
            #         target = self.X[indices]                            
            #         for dim in range(self.n):                               
            #             self.centroids[dim] = target[:,dim].mean()
            
            for k in range(self.K):
                indices = [i for i in range(len(self.clusters)) if self.clusters[i] == k]
                target = self.X[indices]
                if(indices != []): 
                    for dim in range(self.n):                               
                            self.centroids[k][dim] = target[:,dim].mean()
        return self.clusters,self.centroids
def initialize_centroids(dim):
    a = []
    random.seed()
    for i in range(dim):
        temp = random.random()
        a.append(temp)
    return a