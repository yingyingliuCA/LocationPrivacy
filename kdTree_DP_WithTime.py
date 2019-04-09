# KDtree Source: https://www.astroml.org/book_figures/chapter2/fig_kdtree_example.html
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from scipy import stats

#------------------------------------------------------------
# Preprocessing: read file, remove outliers, aggregate by unique users

data = pandas.read_csv('loc-gowalla_totalCheckins.txt',sep="\t",header=None,encoding='utf-8').fillna(0)
attr1 = 2
attr2 = 3 
attr3 = 1 
scaler1 = MinMaxScaler(feature_range=(0,1)).fit(data[attr1].values.reshape(-1,1))
longtitude = list(scaler1.transform(data[attr1].values.reshape(-1,1)).reshape(1,-1)[0])
scaler2 = MinMaxScaler(feature_range=(0,1)).fit(data[attr2].values.reshape(-1,1))
latitude = list(scaler2.transform(data[attr2].values.reshape(-1,1)).reshape(1,-1)[0])
timestampstr = list(map(str,data[attr3].values))

#there are some outliers that greatly affect kd-tree, remove them
longlat = list(zip(longtitude,latitude))
longlat = [list(item) for item in longlat]
z = np.abs(stats.zscore(longlat))
out = list(np.where(z>6)[0]) #37 outliers. if z>5, then zout[0] has 56010 items

#try the naive approach where longtitude > 0.5 by observing the kd-tree graph before outliers are removed
#only 28 elements with longtitude > 0.5
# out = [i for i,x in enumerate(longtitude) if x>0.5]

#this is not very efficient. Need to improve
longtitude2 = []
latitude2 = []
timestampstr2 = []
for i in range(len(longtitude)):
    # if i not in (zout[0]):
    if i not in out: 
        longtitude2.append(longtitude[i])
        latitude2.append(latitude[i])
        timestampstr2.append(timestampstr[i])

longlat2 = list(zip(longtitude2,latitude2))
longlat2 = [list(item) for item in longlat2]
scaler3 = MinMaxScaler(feature_range=(0,1)).fit(longlat2)
longlatlist = list(scaler3.transform(longlat2))

longstrDecimal = []
latstrDecimal = []
for i in range(len(longtitude2)):
    # longstrDecimal.append("{:.5f}".format(longtitude[i])) #keep trailing zero's
    longstrDecimal.append("{:.5f}".format(longlatlist[i][0]))
for i in range(len(latitude2)):
    # latstrDecimal.append("{:.5f}".format(latitude[i]))    
    latstrDecimal.append("{:.5f}".format(longlatlist[i][1]))  

locNormalized = [x+','+y for x,y in zip(longstrDecimal,latstrDecimal)]
# datestr = [x[0:10] for x in timestampstr]
datestr = [x[0:10] for x in timestampstr2]
locdate = [x+y for x,y in zip(locNormalized,datestr)]
locdate = list(dict.fromkeys(locdate)) #remove duplicates
locdate.sort()
locNormalizedUniqueUsers = [x[0:15] for x in locdate]
# dateUniqueUsers = [x[9:] for x in locdate]

dateUniqueUsers = []
for i in range(len(locdate)):
    dateUniqueUsers.append(datetime.strptime(locdate[i][15:],'%Y-%m-%d'))

# uniqueDates = Counter(dateUniqueUsers)
minDate = min(dateUniqueUsers)
maxDate = max(dateUniqueUsers)
uniqueLocNormalized = Counter(locNormalizedUniqueUsers)

#LocArray for k-d tree
LocArray=[]
for x in uniqueLocNormalized:
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
# KDTSearchHeight = 6 
# epsilon = 0.5

KDT = KDTree(LocArray, [-1.1, -0.1], [1.1, 1.1])

# for KDTSearchHeight in range (4, 7):
for KDTSearchHeight in range (6, 7):

    #Generalize locations to groups according to the cut by KD tree
    LocGroups=[]
    KDT.find_groups(KDTSearchHeight, LocGroups)
    print('KDTSearchHeight=',KDTSearchHeight,'Number of Location groups:',len(LocGroups))

    # #------------------------------------------------------------
    # # Plot four different levels of the KD tree
    # #fig = plt.figure(figsize=(5, 5))
    # fig = plt.figure(figsize=(6, 6))

    # # fig.subplots_adjust(wspace=0.1, hspace=0.15,
    # #                     left=0.1, right=0.9,
    # #                     bottom=0.05, top=0.9)

    # #for level in range(1, 5):
    # # ax = fig.add_subplot(2, 2, level, xticks=[], yticks=[])
    # #ax.scatter(X[:, 0], X[:, 1], s=9)
    # #Ying Ying Liu
    # # level = KDTSearchHeight + 1
    # ax = fig.add_subplot(1, 1, 1, xticks=[0,1], yticks=[0,1])
    # ax.scatter(LocArray[:, 0], LocArray[:, 1], s=9)
    # KDT.draw_rectangle(ax, depth=KDTSearchHeight)

    # # ax.set_xlim(-1.2, 1.2)
    # # ax.set_ylim(-0.15, 1.15)
    # #ax.set_title('level %i' % level)

    # # suptitle() adds a title to the entire figure
    # #fig.suptitle('$k$d-tree Example')
    # #fig.suptitle('$k$d-tree for Gowalla Locations of 1000 Users')
    # # plt.title('$k-$d tree of level %i for Gowalla Locations of 1000 Users' % level)
    # plt.title('$k-$d tree of depth %i for Gowalla Locations' %KDTSearchHeight)
    # figname = 'GowallaUsers_kdTree_'+str(loopcount)+'.png'
    # plt.savefig(figname)
    # # plt.show()   
    # #print(groups)

    GroupListForLocDate = []
    GroupDict = {}
    DateGroupDictWithUserCounts = {}

    #Aggregate count of users based on location groups and date
    for i in range(len(LocGroups)):
        arrays = LocGroups[i]
        for j in range(len(arrays)):
            TmpList = arrays[j].tolist()
            # GroupDict[','.join(map(str, TmpList))]=i
            longDecimal = "{:.5f}".format(TmpList[0])
            latDecimal = "{:.5f}".format(TmpList[1])
            GroupDict[longDecimal+','+latDecimal] = i 

    for i in range(len(locdate)):
        LocStr = locdate[i][0:15]
        DateStr = datetime.strptime(locdate[i][15:],'%Y-%m-%d')
        LocGroupStr = GroupDict[LocStr] 
        GroupListForLocDate.append(LocGroupStr)
        if (DateStr,LocGroupStr) not in DateGroupDictWithUserCounts: 
            DateGroupDictWithUserCounts[(DateStr,LocGroupStr)] = 1
        else:
            DateGroupDictWithUserCounts[(DateStr,LocGroupStr)] = DateGroupDictWithUserCounts[(DateStr,LocGroupStr)] + 1

    print('length of DateGroupDictWithUserCounts raw:',len(DateGroupDictWithUserCounts))
    
    #build a contingency table for each date
    #in order for the predictor to work, it is important to have a record for each day
    dateList = [minDate + timedelta(days=x) for x in range(0,abs((maxDate-minDate).days)+1)]

    uniqueGroups = Counter(GroupListForLocDate)
    for x in dateList: 
        for y in uniqueGroups: 
            if (x,y) not in DateGroupDictWithUserCounts: 
                DateGroupDictWithUserCounts[(x,y)] = 0 
    print('length of DateGroupDictWithUserCounts contingency:',len(DateGroupDictWithUserCounts))


    loopcount = 1       
    # for epsilon in ([0.1, 0.5, 1]): 
    for epsilon in ([0.5,1]): 

        print('epsilon=',epsilon)
        DateGroupDictWithUserCountsDP = {}

        #------------------------------------------------------------
        # Differential Privacy

        #Differential privacy - add Laplace noise:
        for keys,value in DateGroupDictWithUserCounts.items():
            CountWithNoise = value + np.random.laplace(scale = 1/epsilon)
            #post processing: round to closest non-negative integer
            if CountWithNoise < 0:
                CountWithNoise = 0
            else:
                CountWithNoise = int(CountWithNoise + 0.5)   
            DateGroupDictWithUserCountsDP[keys] = CountWithNoise

        # file1 = 'loc-gowalla_totalCheckins_DateGroupUserCounts_'+str(loopcount)+'.csv'
        file2 = 'loc-gowalla_totalCheckins_DateGroupUserCountsDP_'+str(loopcount)+'.csv'

        # with open(file1,"w") as f:
        #     for keys,value in DateGroupDictWithUserCounts.items():
        #         f.write("%s,%s,%s\n"%(keys[0],keys[1],value))
        with open(file2,"w") as f:
            for keys,value in DateGroupDictWithUserCountsDP.items():
                f.write("%s,%s,%s\n"%(keys[0],keys[1],value))

        loopcount = loopcount + 1



