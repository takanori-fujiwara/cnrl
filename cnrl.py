import copy
import numpy as np
from sklearn import preprocessing


class CNRL():
    '''
    cNRL from Fujiwara et al., 2020 (arXiv:2005.12419).

    Parameters
    ----------
    nw_learner: network representation learning object
        A network representation learning object for the network representation
        learning step. nw_learner must have fit, transform, get_feat_defs
        methods, such as DeepGL in https://github.com/takanori-fujiwara/deepgl
    contrast_learner: contrastive learning object
        A contrastive learning object for the contrastive learning step.
        contrast_learner must have fit, transform, get_loadings(), and
        get_components(), such as cPCA and ccPCA in
        https://github.com/takanori-fujiwara/ccpca.
    thres_corr_cl_feats: float, optional, (default=0)
        Threshold to pruce network features when performing contrastive lerning.
        The network features that have higher Pearson correlation coefficient
        than thres_corr_cl_feats with another network feature will be pruned.
    scaling_cl_inputs: boolean, optional, (default=True)
        If true, applying standardization to input matrices of contrast_learner
        before performing the contrastive learning step.
    Attributes
    ----------
    tg_feat_mat: ndarray, shape(n_nodes, n_network_features)
        Tareget network's feature matrix learned by nw_learner.
    bg_feat_mat: ndarray, shape(n_nodes, n_network_features)
        Background network's feature matrix learned by nw_learner.
    feat_defs: list of strings
        Learned features' definitions by nw_learner
    loadings: ndarray, shape(n_network_features, n_components)
        Loadings to each component learned by contrast_learner.
    components: ndarray, shape(n_network_features, n_components)
        Components/projection matrix learned by contrast_learner.
    nw_learner: network representation learning object
        Access to the input parameter.
    contrast_learner: contrastive learning object
        Access to the input parameter.
    thres_corr_cl_feats: float
        Access to the input parameter.
    scaling_cl_inputs: boolean
        Access to the input parameter.
    Examples
    --------
    >>> # To run this sample analysis, install DeepGL and cPCA first
    >>> # 1. DeepGL: https://github.com/takanori-fujiwara/deepgl
    >>> # 2. cPCA: https://github.com/takanori-fujiwara/ccpca

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import graph_tool.all as gt
    >>> from sklearn import preprocessing

    >>> from cnrl import CNRL
    >>> from deepgl import DeepGL
    >>> from cpca import CPCA

    >>> # Load data. Refer to http://www-personal.umich.edu/~mejn/netdata/ for the details of datasets
    >>> tg = gt.load_graph('./data/dolphin.xml.gz')
    >>> bg = gt.load_graph('./data/karate.xml.gz')

    >>> # Prepare network representation learning method
    >>> nrl = DeepGL(base_feat_defs=[
    ...     'total_degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank',
    ...     'katz'
    ... ],
    ...              rel_feat_ops=['mean', 'sum', 'maximum', 'lp_norm'],
    ...              nbr_types=['all'],
    ...              ego_dist=3,
    ...              lambda_value=0.7)

    >>> # Prepare contrastive learning method
    >>> cl = CPCA()

    >>> # Set network representation and contrastive learning methods
    >>> # using DeepGL and cPCA is i-cNRL (interpretable cNRL)
    >>> cnrl = CNRL(nrl, cl)

    >>> # Learning
    >>> cnrl.fit(tg, bg)

    >>> # Obtain results for plotting
    >>> tg_feat_mat = preprocessing.scale(cnrl.tg_feat_mat)
    >>> bg_feat_mat = preprocessing.scale(cnrl.bg_feat_mat)
    >>> tg_emb = cnrl.transform(feat_mat=tg_feat_mat)
    >>> bg_emb = cnrl.transform(feat_mat=bg_feat_mat)

    >>> feat_defs = cnrl.feat_defs
    >>> fc_cpca = cnrl.loadings
    >>> fc_cpca_max = abs(fc_cpca).max()
    >>> if fc_cpca_max == 0:
    ...     fc_cpca_max = 1
    >>> fc_cpca /= fc_cpca_max

    >>> # Plot
    >>> # Plot 1: Embedding result
    >>> plt.figure(figsize=(6, 6))
    >>> plt.scatter(tg_emb[:, 0], tg_emb[:, 1], c='orange', s=10)
    >>> plt.scatter(bg_emb[:, 0], bg_emb[:, 1], c='green', s=10)
    >>> plt.legend(['target', 'background'])
    >>> plt.title("cPCA")

    >>> # Plot 2: feature contributions
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> im = ax.imshow(fc_cpca, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    >>> # plot feature names
    >>> ax.set_yticks(np.arange(len(feat_defs)))
    >>> ax.yaxis.tick_right()
    >>> ax.set_yticklabels(feat_defs, fontsize=12)
    >>> # plot col names
    >>> xlabel_names = ["cPC 1", "cPC 2"]
    >>> ax.set_xticks(np.arange(len(xlabel_names)))
    >>> ax.set_xticklabels(xlabel_names)
    >>> xlbls = ax.get_xticklabels()
    >>> plt.setp(xlbls)
    >>> plt.tight_layout()

    >>> plt.show()
    '''
    def __init__(self,
                 nw_learner,
                 contrast_learner,
                 thres_corr_cl_feats=0,
                 scaling_cl_inputs=True):
        self.nw_learner = nw_learner
        self.contrast_learner = contrast_learner
        self.thres_corr_cl_feats = thres_corr_cl_feats
        self.scaling_cl_inputs = scaling_cl_inputs
        self.tg_feat_mat = None
        self.bg_feat_mat = None
        self.feat_defs = None
        self.loadings = None
        self.components = None

    def fit_transform(self, tg_nw, bg_nw, **contrast_learner_kwargs):
        '''
        Apply fit ant transform to target and background networks.

        Parameters
        ----------
        tg_nw: network object
            Target network that can be handled by nw_learner. For example,
            if nw_learner is DeepGL (https://github.com/takanori-fujiwara/deepgl),
            tg_nw must be a graph-tool's graph object.
        bg_nw: network object
            Background network that can be handled by nw_learner.
        contrast_learner_kwargs: keyword arguments
            Keyword arguments used for contrast_learner. For example, if
            contrast_learner is cPCA (https://github.com/takanori-fujiwara/ccpca),
            auto_alpha_selection=False, alpha=100, etc. can be used.
        Return
        ----------
        embedding_result: ndarray, shape(n_nodes, n_components)
            Contrastive embedding result.
        '''
        self.learn_nw_repr(tg_nw, bg_nw)
        self.learn_contrast(self.tg_feat_mat, self.bg_feat_mat, self.feat_defs,
                            **contrast_learner_kwargs)
        result = None
        if self.contrast_learner.__class__.__name__ == 'CCPCA':
            concat_mat = np.vstack((self.tg_feat_mat, self.bg_feat_mat))
            result = self.transform(feat_mat=concat_mat)
        else:
            result = self.transform(feat_mat=self.tg_feat_mat)

        return result

    def fit(self, tg_nw, bg_nw, **contrast_learner_kwargs):
        '''
        Apply fit  to target and background networks.

        Parameters
        ----------
        tg_nw: network object
            Target network that can be handled by nw_learner. For example,
            if nw_learner is DeepGL (https://github.com/takanori-fujiwara/deepgl),
            tg_nw must be a graph-tool's graph object.
        bg_nw: network object
            Background network that can be handled by nw_learner.
        contrast_learner_kwargs: keyword arguments
            Keyword arguments used for contrast_learner. For example, if
            contrast_learner is cPCA (https://github.com/takanori-fujiwara/ccpca),
            auto_alpha_selection=False, alpha=100, etc. can be used.
        Return
        ----------
        self
        '''
        self.learn_nw_repr(tg_nw, bg_nw)
        self.learn_contrast(self.tg_feat_mat, self.bg_feat_mat, self.feat_defs,
                            **contrast_learner_kwargs)

        return self

    def transform(self, feat_mat=None, nw=None):
        '''
        Apply transform to a network or a network feature matrix

        Parameters
        ----------
        feat_mat: array_like, shape(n_node, n_network_features), optional (default=None)
            Network feature matrix that contains the same feature set with the
            learned feature matrix by fitting with nw_learner. If None, produce
            a network feature matrix from nw (input network) and then use it as
            feat_mat.
        nw: network object
            If feat_mat is None and nw is not None, produce a network feature
            matrix from nw and then use it as feat_mat.
        Return
        ----------
        embedding_result: ndarray, shape(n_nodes, n_components)
            Contrastive embedding result.
        '''
        result = None

        if feat_mat is None and nw is not None:
            feat_mat = self.nw_learner.transform(nw,
                                                 diffusion_iter=0,
                                                 transform_method=None)

        result = self.contrast_learner.transform(feat_mat)
        return result

    def learn_nw_repr(self, tg_nw, bg_nw):
        '''
        Apply network representation learning.

        Parameters
        ----------
        tg_nw: network object
            Target network that can be handled by nw_learner. For example,
            if nw_learner is DeepGL (https://github.com/takanori-fujiwara/deepgl),
            tg_nw must be a graph-tool's graph object.
        bg_nw: network object
            Background network that can be handled by nw_learner.
        Return
        ----------
        self
        '''
        self.nw_learner.fit(tg_nw)
        self.tg_feat_mat = self.nw_learner.transform(tg_nw,
                                                     diffusion_iter=0,
                                                     transform_method=None)
        self.bg_feat_mat = self.nw_learner.transform(bg_nw,
                                                     diffusion_iter=0,
                                                     transform_method=None)
        self.feat_defs = self.nw_learner.get_feat_defs()

        return self

    def learn_contrast(self, tg_feat_mat, bg_feat_mat, feat_defs, **kwargs):
        '''
        Apply contrastive learning.

        Parameters
        ----------
        tg_feat_mat: array_like, shape(n_nodes, n_network_features)
            Target network's feature matrix obtained via network representation
            learning.
        bg_feat_mat: array_like, shape(n_nodes, n_network_features)
            Target network's feature matrix obtained via network representation
            learning.
        feat_defs: list of strings
            Learned features' definitions by network representation learning.
        kwargs: keyword arguments
            Keyword arguments used for contrast_learner. For example, if
            contrast_learner is cPCA (https://github.com/takanori-fujiwara/ccpca),
            auto_alpha_selection=False, alpha=100, etc. can be used.
        Return
        ----------
        self
        '''
        if self.thres_corr_cl_feats > 0:
            corr_mat = np.corrcoef(tg_feat_mat, rowvar=False)
            np.fill_diagonal(corr_mat, 0.0)
            high_correlated = abs(corr_mat) > self.thres_corr_cl_feats
            n_feats = high_correlated.shape[0]
            keeping_feat_indices = [True for _ in range(n_feats)]

            for i in range(n_feats):
                for j in range(i, n_feats):
                    if high_correlated[i, j] == True:
                        keeping_feat_indices[j] = False

            tg_feat_mat = tg_feat_mat[:, keeping_feat_indices]
            bg_feat_mat = bg_feat_mat[:, keeping_feat_indices]
            feat_defs = list(np.array(feat_defs)[keeping_feat_indices])
            # TODO: avoid using copy
            self.tg_feat_mat = np.copy(tg_feat_mat)
            self.bg_feat_mat = np.copy(bg_feat_mat)
            self.feat_defs = copy.deepcopy(feat_defs)

        if self.scaling_cl_inputs:
            self.contrast_learner.fit(preprocessing.scale(tg_feat_mat),
                                      preprocessing.scale(bg_feat_mat),
                                      **kwargs)
        else:
            self.contrast_learner.fit(tg_feat_mat, bg_feat_mat, **kwargs)
        self.loadings = self.contrast_learner.get_loadings()
        self.components = self.contrast_learner.get_components()

        return self

    def set_nw_learner(self, nw_learner):
        '''
        Set a network representation learning method

        Parameters
        ----------
        nw_learner: network representation learning object
            A network representation learning object for the network representation
            learning step. nw_learner must have fit, transform, get_feat_defs
            methods, such as DeepGL in https://github.com/takanori-fujiwara/deepgl
        Return
        ----------
        self
        '''
        self.nw_learner = nw_learner
        return self

    def set_contrast_learner(self, contrast_learner):
        '''
        Set a contrastive learning method

        Parameters
        ----------
        contrast_learner: contrastive learning object
            A contrastive learning object for the contrastive learning step.
            contrast_learner must have fit, transform, get_loadings(), and
            get_components(), such as cPCA and ccPCA in
            https://github.com/takanori-fujiwara/ccpca.
        Return
        ----------
        self
        '''
        self.contrast_learner = contrast_learner
        return self
