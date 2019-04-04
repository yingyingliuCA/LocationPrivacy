# Source: https://www.astroml.org/book_figures/chapter2/fig_kdtree_example.html
# Retrieved: 2019-04-02

# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib import pyplot as plt

# Revision history: 
# Ying Ying Liu, 2019-04-03
# 1. Added find_groups 
# 2. Changed visualization
# 3. Also added Differential privacy code here
# todo: properly organize the code

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=8, usetex=True)
setup_text_plots(fontsize=8, usetex=False) #Ying Ying Liu


# We'll create a KDTree class which will recursively subdivide the
# space into rectangular regions.  Note that this is just an example
# and shouldn't be used for real computation; instead use the optimized
# code in scipy.spatial.cKDTree or sklearn.neighbors.BallTree
class KDTree:
    """Simple KD tree class"""

    # class initialization function
    def __init__(self, data, mins, maxs):
        self.data = np.asarray(data)

        # data should be two-dimensional
        assert self.data.shape[1] == 2

        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.child1 = None
        self.child2 = None

        if len(data) > 1:
            # sort on the dimension with the largest spread
            largest_dim = np.argmax(self.sizes)
            i_sort = np.argsort(self.data[:, largest_dim])
            self.data[:] = self.data[i_sort, :]

            # find split point
            N = self.data.shape[0]
            split_point = 0.5 * (self.data[int(N / 2), largest_dim]
                                 + self.data[int(N / 2) - 1, largest_dim])

            # create subnodes
            mins1 = self.mins.copy()
            mins1[largest_dim] = split_point
            maxs2 = self.maxs.copy()
            maxs2[largest_dim] = split_point

            # Recursively build a KD-tree on each sub-node
            self.child1 = KDTree(self.data[int(N / 2):], mins1, self.maxs)
            self.child2 = KDTree(self.data[:int(N / 2)], self.mins, maxs2)

    def draw_rectangle(self, ax, depth=None):
        """Recursively plot a visualization of the KD tree region"""
        if depth == 0:
            rect = plt.Rectangle(self.mins, *self.sizes, ec='k', fc='none')
            ax.add_patch(rect)

        if self.child1 is not None:
            if depth is None:
                self.child1.draw_rectangle(ax)
                self.child2.draw_rectangle(ax)
            elif depth > 0:
                self.child1.draw_rectangle(ax, depth - 1)
                self.child2.draw_rectangle(ax, depth - 1)

    #Ying Ying Liu
    def find_groups(self, depth, groups):
        if depth == 0:
            groups.append(self.data)

        if self.child1 is not None:
            if depth > 0:
                self.child1.find_groups(depth-1,groups)
                self.child2.find_groups(depth-1,groups)


#------------------------------------------------------------
# Create a set of structured random points in two dimensions
# np.random.seed(0)

# X = np.random.random((30, 2)) * 2 - 1
# X[:, 1] *= 0.1
# X[:, 1] += X[:, 0] ** 2

# Ying Ying Liu: read in the location from the csv
import csv
import numpy as np
import pandas
from collections import Counter

colnames = ['UserID','NormalizedUniqueLocation']
data = pandas.read_csv('loc-gowalla_totalCheckins_1000_LocAndUserRaw.csv',names=colnames)
LocList = data.NormalizedUniqueLocation.tolist()
del LocList[0] #remove header
UniqueLoc = Counter(LocList)
LocArray = []
for x in UniqueLoc:
    TmpArray = list(map(float,x.split(",")))
    LocArray.append(TmpArray)
LocArray = np.asarray(LocArray)


#------------------------------------------------------------
# Use our KD Tree class to recursively divide the space
#KDT = KDTree(X, [-1.1, -0.1], [1.1, 1.1])

#------------------------------------------------------------
# KD Tree + DP 
#Ying Ying Liu

#Experiments: change these two parameters
KDTSearchHeight = 7 
epsilon = 0.1

#Generalize locations to groups according to the cut by KD tree
KDT = KDTree(LocArray, [-1.1, -0.1], [1.1, 1.1])
LocGroups=[]
KDT.find_groups(KDTSearchHeight, LocGroups)
#print(groups)
print("Number of Location groups:",len(LocGroups))

GroupListForUserLoc = []
LocGroupWithUserCount = []
LocGroupWithUserCountDP = []
GroupDict = {}

#Aggregate count of users based on location groups
for i in range(len(LocGroups)):
    arrays = LocGroups[i]
    for j in range(len(arrays)):
        TmpList = arrays[j].tolist()
        GroupDict[','.join(map(str, TmpList))]=i
# print(GroupDict)

for i in range(len(LocList)):
    #this can get rid of the trailing 0's
    TmpList = list(map(float,LocList[i].split(",")))
    TmpString = ','.join(map(str, TmpList))
    TmpList2 = [LocList[i]]
    TmpList2.append(GroupDict[TmpString])
    GroupListForUserLoc.append(GroupDict[TmpString])

UniqueGroups = Counter(GroupListForUserLoc)
for i in range(len(LocGroups)):
    TmpList = [i,UniqueGroups[i]]
    LocGroupWithUserCount.append(TmpList)

#Differential privacy - add Laplace noise:
for i in range(len(LocGroups)):
    CountWithNoise = UniqueGroups[i] + np.random.laplace(scale = 1/epsilon)
    #post processing: round to closest non-negative integer
    if CountWithNoise < 0:
        CountWithNoise = 0
    else:
        CountWithNoise = int(CountWithNoise + 0.5)   
    TmpList = [i,CountWithNoise]
    LocGroupWithUserCountDP.append(TmpList)

with open('loc-gowalla_totalCheckins_1000_LocWithGroupsRaw.csv',"w") as f:
    writer = csv.writer(f)
    for i in range(len(LocGroupWithUserCount)):
        writer.writerow(LocGroupWithUserCount[i])
with open('loc-gowalla_totalCheckins_1000_LocWithGroupsDP.csv',"w") as f:
    writer = csv.writer(f)
    for i in range(len(LocGroupWithUserCountDP)):
        writer.writerow(LocGroupWithUserCountDP[i])


# with open('loc-gowalla_totalCheckins_1000_LocGroups.csv',"w") as f:
#     writer = csv.writer(f)
#     for i in range(len(groups)):
#         arrays = groups[i]
#         for j in range(len(arrays)):
#             TmpList = arrays[j].tolist()
#             TmpList.append(i)
#             writer.writerow(TmpList)


#------------------------------------------------------------
# Plot four different levels of the KD tree
#fig = plt.figure(figsize=(5, 5))
fig = plt.figure(figsize=(6, 6))

# fig.subplots_adjust(wspace=0.1, hspace=0.15,
#                     left=0.1, right=0.9,
#                     bottom=0.05, top=0.9)

#for level in range(1, 5):
# ax = fig.add_subplot(2, 2, level, xticks=[], yticks=[])
#ax.scatter(X[:, 0], X[:, 1], s=9)
#Ying Ying Liu
level = KDTSearchHeight + 1
ax = fig.add_subplot(1, 1, 1, xticks=[0,1], yticks=[0,1])
ax.scatter(LocArray[:, 0], LocArray[:, 1], s=9)
KDT.draw_rectangle(ax, depth=level - 1)

# ax.set_xlim(-1.2, 1.2)
# ax.set_ylim(-0.15, 1.15)
#ax.set_title('level %i' % level)

# suptitle() adds a title to the entire figure
#fig.suptitle('$k$d-tree Example')
#fig.suptitle('$k$d-tree for Gowalla Locations of 1000 Users')
plt.title('$k-$d tree of level %i for Gowalla Locations of 1000 Users' % level)
plt.savefig('GowallaUsers0_1000_kdTree.png')
plt.show()



