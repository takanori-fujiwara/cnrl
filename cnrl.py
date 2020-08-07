import copy
import numpy as np
from sklearn import preprocessing


class CNRL():
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
        self.learn_nw_repr(tg_nw, bg_nw)
        self.learn_contrast(self.tg_feat_mat, self.bg_feat_mat, self.feat_defs,
                            **contrast_learner_kwargs)

        return self

    def transform(self, feat_mat=None, nw=None):
        result = None

        if feat_mat is None and nw is not None:
            feat_mat = self.nw_learner.transform(nw,
                                                 diffusion_iter=0,
                                                 transform_method=None)

        result = self.contrast_learner.transform(feat_mat)
        return result

    def learn_nw_repr(self, tg_nw, bg_nw):
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
        self.nw_learner = nw_learner
        return self

    def set_contrast_learner(self, nw_learner):
        self.contrast_learner = contrast_learner
        return self
