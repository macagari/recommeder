import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from dao.general_dao import Dao, orders_to_bidict


class ImplicitFeedbackPreprocessing:

    def __init__(self,
                 collection_name: str,
                 df: pd.DataFrame = None,
                 alpha: float = 40.0):

        self.dao = Dao(collection_name)

        if df is None:  # TODO: not need? replace with collection name
            self.df = self.dao.get_df()
        else:
            self.df = df
        self.alpha = alpha

        self.users = list(self.df['users'].unique())

        self.items = list(self.df['items'].unique())
        #print(self.items)
        self.n_users = len(orders_to_bidict(collection_name)['users'])
        self.n_items = len(orders_to_bidict(collection_name)['items'])
        print("n_items",self.n_items)
        print("n_users", self.n_users)
        self.id2data = orders_to_bidict(collection_name)

        self.user_index = np.array(self.df['users'].apply(self.dao.user2id))
        # TODO: we can admit just items that are in the collection with pictures
        self.item_index = np.array(self.df['items'].apply(self.dao.item2id))

    def sparse_matrix(self):
        sparse_ui = csr_matrix(([self.alpha] * len(self.df),
                                (self.user_index, self.item_index)),
                               shape=(self.n_users, self.n_items))
        sparse_iu = sparse_ui.T.tocsr()

        return sparse_ui, sparse_iu




if __name__ == '__main__':
    pass
    # d = ImplicitFeedbackPreprocessing(collection_name="luisa_spagnoli")
    # print(d.n_users)
    # print(len(d.users))
    # print(d.item_index)




