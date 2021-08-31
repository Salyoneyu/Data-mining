import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
# https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
# https://scikit-learn.org/0.15/modules/generated/sklearn.base.ClassifierMixin.html
import networkx as nx
# nx 1.9
import pandas as pd

#定义高斯核，更加平滑的选择
def kernel_gaussian(x, y, h, degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel

#x(t+1)，也就是当前向量xt更新后的值
def step(x_l0, X, W=None, h=0.1):
    # 选择数据，前两个属性
    n = X.shape[0]
    d = X.shape[1]
    # 核矩阵权值，也就是影响值
    superweight = 0.
    # 首先生成一行d维的0矩阵
    x_t1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1))
    else:
        W = W
    #计算核，带权的核函数的加权平均
    for j in range(n):
        kernel = kernel_gaussian(x_l0, X[j], h, d)
        kernel = kernel * W[j] / (h ** d)
        superweight = superweight + kernel
        # 计算x（t+1）
        x_t1 = x_t1 + (kernel * X[j])
    x_t1 = x_t1 / superweight
    density = superweight / np.sum(W)
    return [x_t1, density]


#求梯度变化最大的地方最容易发现极值，梯度上升过程
# 设置参数，数据集，eps最小密度阈值，窗口宽度h
def climb(x_t, X, W=None, h=0.1, eps=1e-4):
    radius_0 = 0.
    radius_1 = 0.
    radius_2 = 0.
    iters = 0.
    # 概率
    prob = 0.
    x_t1 = np.copy(x_t)
    while True:
        radius_3 = radius_2
        radius_2 = radius_1
        radius_1 = radius_0
        x_l0 = np.copy(x_t1)       #X(t)
        x_t1, density = step(x_l0, X, W=W, h=h)
        error = density - prob
        prob = density
        radius_0 = np.linalg.norm(x_t1 - x_l0)
        radius = radius_3 + radius_2 + radius_1 + radius_0
        iters += 1
        if iters > 3 and error < eps:
            break
    return [x_t1, prob, radius]

# 变化最大的地方最容易发现极值
class DENCLUE(BaseEstimator, ClusterMixin):
    def __init__(self, h=None, eps=1e-8, min_density=0., metric='euclidean'):
        self.h = h
        self.eps = eps
        self.min_density = min_density
        self.metric = metric
    # 分簇
    def classify(self, X, y=None, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps必须是正定的.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        density_attractors = np.zeros((self.n_samples, self.n_features))
        rad = np.zeros((self.n_samples, 1))
        density = np.zeros((self.n_samples, 1))

        # 初始化，所有的点数据都是噪音
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1))
        else:
            sample_weight = sample_weight
        labels = -np.ones(X.shape[0])
        # 密度吸引子
        for i in range(self.n_samples):
            density_attractors[i], density[i], rad[i] = climb(X[i], X, W=sample_weight,
                                                                      h=self.h, eps=self.eps)
        # 准备输出格式
        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'set_cluster': [0],
                                      'centroid': np.atleast_2d(density_attractors[0])}
        g_clusters = nx.Graph()
        for j1 in range(self.n_samples):
            g_clusters.add_node(j1, attr_dict={'attractor': density_attractors[j1], 'radius': rad[j1],
                                               'cluster_purity': density[j1]})
        # 聚类构造
        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):
                if g_clusters.has_edge(j1, j2):
                    continue
                diff = np.linalg.norm(g_clusters.node[j1]['attractor'] - g_clusters.node[j2]['attractor'])
                if diff <= (g_clusters.node[j1]['radius'] + g_clusters.node[j1]['radius']):
                    g_clusters.add_edge(j1, j2)
        clusters = list(nx.connected_component_subgraphs(g_clusters))
        num_clusters = 0

        # 组织好聚类以便输出
        for clust in clusters:
            # 找到密度吸引子的最大密度及对应的数据信息
            max_instance = max(clust, key=lambda x: clust.node[x]['cluster_purity'])
            max_density = clust.node[max_instance]['cluster_purity']
            max_centroid = clust.node[max_instance]['attractor']
            complete = False
            c_size = len(clust.nodes())
            if clust.number_of_edges() == (c_size * (c_size - 1)) / 2.:
                complete = True
            cluster_info[num_clusters] = {'set_cluster': clust.nodes(),
                                          'size': c_size,
                                          'centroid': max_centroid,
                                          'cluster_purity': max_density}
            # 判断噪音点
            if max_density >= self.min_density:
                labels[clust.nodes()] = num_clusters
            num_clusters += 1
        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self


data = pd.read_csv('iris1.csv')
data = np.array(data)
samples = np.mat(data[:,0:2])
true_labels=data[:,-1]
labels=list(set(true_labels))
true_ID=np.zeros((3,50))
index=range(len(true_labels))
for i in range(len(labels)):
    true_ID[i]=[j for j in index if true_labels[j]==labels[i]]
d = DENCLUE(0.25, 0.0001)
d.classify(samples)
right_num=0

for i in range(len(d.clust_info_)):
    bestlens=0
    clust_set = set(d.clust_info_[i]['set_cluster'])
    for j in range(len(labels)):
        true_set=set(true_ID[j])
        and_set= clust_set&true_set
        if len(list(and_set))>bestlens:
            bestlens=len(list(and_set))
    right_num+=bestlens

print(d.clust_info_,float(right_num)/len(samples))

