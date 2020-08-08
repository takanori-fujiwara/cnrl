import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
from sklearn import preprocessing

from cnrl import CNRL
from deepgl import DeepGL
from cpca import CPCA

# To run this sample analysis, install DeepGL and cPCA first
# 1. DeepGL: https://github.com/takanori-fujiwara/deepgl
# 2. cPCA: https://github.com/takanori-fujiwara/ccpca

# Load data. Refer to http://www-personal.umich.edu/~mejn/netdata/ for the details of datasets
tg = gt.load_graph('./data/dolphin.xml.gz')
bg = gt.load_graph('./data/karate.xml.gz')

# Prepare network representation learning method
nrl = DeepGL(base_feat_defs=[
    'total_degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank',
    'katz'
],
             rel_feat_ops=['mean', 'sum', 'maximum', 'lp_norm'],
             nbr_types=['all'],
             ego_dist=3,
             lambda_value=0.7)

# Prepare contrastive learning method
cl = CPCA()

# Set network representation and contrastive learning methods
# using DeepGL and cPCA is i-cNRL (interpretable cNRL)
cnrl = CNRL(nrl, cl)

# Learning
cnrl.fit(tg, bg)

# Obtain results for plotting
tg_feat_mat = preprocessing.scale(cnrl.tg_feat_mat)
bg_feat_mat = preprocessing.scale(cnrl.bg_feat_mat)
tg_emb = cnrl.transform(feat_mat=tg_feat_mat)
bg_emb = cnrl.transform(feat_mat=bg_feat_mat)

feat_defs = cnrl.feat_defs
fc_cpca = cnrl.loadings
fc_cpca_max = abs(fc_cpca).max()
if fc_cpca_max == 0:
    fc_cpca_max = 1
fc_cpca /= fc_cpca_max

# Plot
# Plot 1: Embedding result
plt.figure(figsize=(6, 6))
plt.scatter(tg_emb[:, 0], tg_emb[:, 1], c='orange', s=10)
plt.scatter(bg_emb[:, 0], bg_emb[:, 1], c='green', s=10)
plt.legend(['target', 'background'])
plt.title("cPCA")

# Plot 2: feature contributions
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(fc_cpca, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
# plot feature names
ax.set_yticks(np.arange(len(feat_defs)))
ax.yaxis.tick_right()
ax.set_yticklabels(feat_defs, fontsize=12)
# plot col names
xlabel_names = ["cPC 1", "cPC 2"]
ax.set_xticks(np.arange(len(xlabel_names)))
ax.set_xticklabels(xlabel_names)
xlbls = ax.get_xticklabels()
plt.setp(xlbls)
plt.tight_layout()

plt.show()
