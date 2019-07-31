import numpy as np
import pandas as pd
import math
import datetime
from scipy import stats
from random import seed, randrange
from math import sqrt


class DecisionTree():
    def __init__(self, Xtr, Xte, ytr, yte, n_features=10, max_depth=8, min_size=1, sample_size=100, n_trees=50):

        self.Xtr = Xtr
        self.Xte = Xte
        self.ytr = ytr
        self.yte = yte
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_trees = n_trees

    # Split a dataset based on a feature and a threshold
    def test_split(self, feature, threshold, group):
        left = group[self.Xtr[group.astype(int),feature]<threshold]
        right = group[np.logical_not(self.Xtr[group.astype(int),feature]<threshold)]
        gini = self.gini_index((left,right))
        return (left, right), gini


    # Calculate the Gini index
    def gini_index(self, groups):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # calculate class proportions
            proportion_0 = sum(self.ytr[group.astype(int)]==[0])/size
            score += proportion_0**2
            proportion_1 = sum(self.ytr[group.astype(int)]==[1])/size
            score += proportion_1**2
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Find best split point for a node
    def find_split(self,group):
        index, value, score, groups = np.inf, np.inf, np.inf, None
        features = np.random.choice(self.Xtr.shape[1], self.n_features, replace=True)
        for feature in features:
            for row_idx in group:
                groups, gini = self.test_split(feature, self.Xtr[row_idx.astype(int),feature.astype(int)],group)
                if gini < score:
                    index, value, score, groups = feature, self.Xtr[row_idx.astype(int),feature.astype(int)], gini, groups
        return {'index':index, 'value':value, 'groups':groups}


    # Create a terminal node value
    def to_terminal(self, group):
        return stats.mode(self.ytr[group.astype(int)])[0]

    # Create child splits for a node or make terminal
    def split(self, node, depth):

        left, right = node['groups']
        del(node['groups'])

        # check for a no split
        if len(left)==0 or len(right)==0:
            node['left'] = node['right'] = self.to_terminal(np.hstack((left,right)))
            return

        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return

        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.find_split(left)
            self.split(node['left'],depth+1)

        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.find_split(right)
            self.split(node['right'], depth+1)


    # Build a decision tree
    def build_tree(self):
        root = self.find_split(self.sample_idx)
        self.split(root, 1)
        return root

    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # Majority voting with list of bagged trees
    def bagging_predict(self, trees, row):
        predictions = np.array([])
        for tree in trees:
            predictions = np.hstack((predictions,self.predict(tree, row)))
        return stats.mode(predictions)[0]

    # Random Forest Algorithm
    def random_forest(self):
        trees = list()
        for i in range(self.n_trees):
            self.sample_idx = np.random.choice(self.Xtr.shape[0], self.sample_size, replace=True)
            tree = self.build_tree()
            trees.append(tree)
            print('Tree: ',i)
            startDT = datetime.datetime.now()
            print (str(startDT))
        predictions = [self.bagging_predict(trees, row) for row in self.Xte]
        return(predictions)
