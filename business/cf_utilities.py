import implicit
import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from scipy.sparse import csr_matrix
from torchvision import models
from torchvision import transforms
from config.config import MODELS
from dao.general_dao import Dao
from models.cf_config import CFConfig


def ui_matrix(data: pd.DataFrame,
              binary: bool = False,
              ) -> np.ndarray:
    n_rows = len(data['users'].unique())
    n_cols = len(data['items'].unique())
    ui_mat = np.zeros((n_rows, n_cols))

    for u, i in zip(data['users'], data['items']):
        ui_mat[u, i] += 1.0

    if binary:
        mask = np.where(ui_mat > 0, 1.0, 0.0)
        return mask

    return ui_mat

def get_cf_model(modelname: str):
    assert modelname in MODELS, 'the model you selected, {}, is not allowed.'.format(modelname)
    cfg = CFConfig()
    if modelname.lower() == 'als':
        return implicit.als.AlternatingLeastSquares(factors=cfg.factors,
                                                    regularization=cfg.regularization,
                                                    dtype=cfg.dtype,
                                                    iterations=cfg.iterations,
                                                    random_state=cfg.random_state)
    elif modelname.lower() == 'lmf':
        return implicit.lmf.LogisticMatrixFactorization(factors=cfg.factors,
                                                        regularization=cfg.regularization,
                                                        dtype=cfg.dtype,
                                                        iterations=cfg.iterations,
                                                        random_state=cfg.random_state)
    elif modelname.lower() == 'bpr':
        return implicit.bpr.BayesianPersonalizedRanking(factors=cfg.factors,
                                                        regularization=cfg.regularization,
                                                        dtype=cfg.dtype,
                                                        iterations=cfg.iterations,
                                                        random_state=cfg.random_state,
                                                        learning_rate=cfg.learning_rate)

class Identity(nn.Module):
    def forward(self, x):
        return x

class ImplicitFeedbackPreprocessing:

    def __init__(self,
                 collection_name: str,
                 df: pd.DataFrame = None,
                 alpha: float = 40.0,
                 ):


        dao = Dao(collection_name=collection_name)
        self.df = dao.get_df()
        self.alpha = alpha
        self.data = dao.dict_users_items

        self.users = list(self.df['users'].unique())
        self.items = list(self.df['items'].unique())
        self.n_users = len(self.data['users'])
        self.n_items = len(self.data['items'])

        self.user_index = np.array(self.df['users'].apply(self.data['users']))
        self.item_index = np.array(self.df['items'].apply(self.data['items']))

    def sparse_matrix(self):
        sparse_ui = csr_matrix(([self.alpha] * len(self.df),
                               (self.user_index, self.item_index)),
                               shape=(self.n_users, self.n_items))
        sparse_iu = sparse_ui.T.tocsr()

        return sparse_ui, sparse_iu

    
