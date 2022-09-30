from abc import ABC
from typing import List, Optional

import pandas as pd
from implicit.lmf import LogisticMatrixFactorization
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

#from metrics.recommender_metrics import recommender_precision, recommender_recall, personalization

from dao.general_dao import Dao
from models.cf_config import CFConfig
from utils.utils_collaborative_filtering import ImplicitFeedbackPreprocessing as IFP
cfg = CFConfig()


class BaseRecommender(ABC):
    def __init__(self,
                 collection_name: str,
                 threshold: Optional[int] = 2,
                 train_months_duration: Optional[int] = 5,
                 alpha: float = 40.0,
                 ):

        self.dao = Dao(collection_name=collection_name)
        self.tot_df, self.train_df, self.test_df = self.dao.temporal_split(threshold=threshold,
                                                                           train_months_duration=train_months_duration)

        self.ifp_tot = IFP(collection_name=collection_name, df=self.tot_df, alpha=alpha)
        self.tot_ui = self.ifp_tot.sparse_matrix()[0]
        self.ifp_train = IFP(collection_name=collection_name, df=self.train_df, alpha=alpha)
        self.train_ui, self.train_iu = self.ifp_train.sparse_matrix()
        self.ifp_test = IFP(collection_name=collection_name, df=self.test_df, alpha=alpha)
        self.test_ui = self.ifp_test.sparse_matrix()[0]
        self.user2prods_test = {}

        for u in self.ifp_test.users:
            self.user2prods_test[self.ifp_test.dao.user2id(u)] = list(
                self.test_df['items'][self.test_df['users'] == u].values)

    # def assess_model(self,
    #                  dict_pred: dict):
    #     k = len(next(iter(dict_pred.values())))
    #     users_metrics = []
    #     tot_rec_len = 0
    #     tot_rel_len = 0
    #     list_predicted = []
    #     list_actual = []
    #
    #     for u in self.ifp_test.users:
    #         user_id = self.dao.user2id(u)
    #         recommended = set(dict_pred.get(user_id))
    #         tot_rec_len += len(recommended)
    #         relevant = set(self.dao.item2id(i) for i in self.user2prods_test.get(user_id))
    #         tot_rel_len += len(relevant)
    #         list_actual.append(list(relevant))
    #         list_predicted.append(list(recommended))
    #         numerator = recommended & relevant
    #         flag_user = 0
    #         if len(numerator) > 0:
    #             flag_user = 1
    #         users_metrics.append(([user_id, len(numerator), flag_user]))
    #
    #     metrics_df = pd.DataFrame(users_metrics,
    #                               columns=['user_id', 'catch', 'flag_user'])
    #     precision = round(recommender_precision(list_predicted, list_actual), 3)
    #     recall = round(recommender_recall(list_predicted, list_actual))
    #     personalization_score = round(personalization(list_predicted), 3)
    #     perc_satisfied_users = round(sum(metrics_df['flag_user']) / metrics_df.shape[0], 3) * 100
    #     print("Precision at k={}: {}".format(k, precision))
    #     print("Recall at k={}: {}".format(k, recall))
    #     print("Personalization score: {}".format(personalization_score))
    #     print("Percentage of satisfied users at k={}: {}".format(k, perc_satisfied_users))

    def fit(self):
        raise NotImplementedError

    # def rank_items(self,
    #                userid: int,
    #                selected_items: List[int]):
    #     raise NotImplementedError

    def recommend(self,
                  userid: int,
                  n: Optional[int] = 10):
        raise NotImplementedError

    # def similar_items(self,
    #                   itemid: int,
    #                   n: Optional[int] = 10):
    #     raise NotImplementedError
    #
    # def similar_users(self,
    #                   userid: int,
    #                   n: Optional[int] = 10):
    #     raise NotImplementedError


class ALS(BaseRecommender):

    def __init__(self,
                 collection_name: str,
                 factors: int = cfg.factors,
                 regularization: float = cfg.regularization,
                 iterations: int = cfg.iterations,
                 threshold: Optional[int] = 2,
                 train_months_duration: int = 5,
                 alpha: float = 40.0):

        super().__init__(collection_name, threshold, train_months_duration, alpha)

        self.model = AlternatingLeastSquares(factors=factors,
                                             regularization=regularization,
                                             iterations=iterations)
        self.fit()

    def fit(self):

        self.model.fit(self.train_iu)
        return self.model

    def recommend(self,
                  username: str,
                  k: Optional[int] = 10,
                  exclude_past: Optional[bool] = True):
        to_remove = None
        if exclude_past:
            to_remove = [self.dao.item2id(el) for el in
                         list(self.train_df.loc[self.train_df['users'] == username, 'items'])]

        user_id = self.dao.user2id(username)

        count_error = 0
        try:
            pre_ = self.model.recommend(userid=user_id,
                                    user_items=self.tot_ui,
                                    N=k,
                                    filter_already_liked_items=exclude_past,
                                    filter_items=to_remove)

        except IndexError as e:
            count_error += 1
            pre_ = []
            print(f'{e=} {user_id=} {count_error=}')

        recommended_items = [self.dao.id2item(idx=p[0]) for p in pre_]
            #print(recommended_items)
        return recommended_items


    def recommend_all_users(self,
                            k: Optional[int] = 10):
        out = {}
        for u in self.ifp_tot.users:
            user_id = self.dao.user2id(username=u)
            out[u] = self.recommend(username=u, k=k)

        return out

class LMF(BaseRecommender):
    def __init__(self,
                 collection_name: str,
                 factors: int = cfg.factors,
                 learning_rate: float = cfg.learning_rate,
                 neg_prop: int = cfg.neg_prop,
                 alpha: float = 1.0,
                 regularization: float = cfg.regularization,
                 iterations: int = cfg.iterations,
                 threshold: int = 2,
                 train_months_duration: int = 5):

        super().__init__(collection_name, threshold=threshold,
                         train_months_duration=train_months_duration,
                         alpha=alpha)

        self.model = LogisticMatrixFactorization(factors=factors,
                                                 learning_rate=learning_rate,
                                                 regularization=regularization,
                                                 iterations=iterations,
                                                 neg_prop=neg_prop)
        self.fit()

    def fit(self):

        self.model.fit(self.train_iu)
        return self.model

    def recommend(self,
                  username: str,
                  k: Optional[int] = 10,
                  exclude_past: Optional[bool] = True):
        to_remove = None
        if exclude_past:
            to_remove = [self.dao.item2id(el) for el in
                         list(self.train_df.loc[self.train_df['users'] == username, 'items'])]

        user_id = self.dao.user2id(username)
        count_error = 0
        try:
            pre_ = self.model.recommend(userid=user_id,
                                    user_items=self.tot_ui,
                                    N=k,
                                    filter_already_liked_items=exclude_past,
                                    filter_items=to_remove)

        except IndexError as e:
            count_error += 1
            pre_ = []
            print(f'{e=} {user_id=} {count_error=}')

        recommended_items = [self.dao.id2item(idx=p[0]) for p in pre_]
        return recommended_items


    def recommend_all_users(self,
                            k: Optional[int] = 10):
        out = {}
        for u in self.ifp_test.users:
            user_id = self.dao.user2id(username=u)
            out[u] = self.recommend(username=u, k=k)

        return out


class BPR(BaseRecommender):
    def __init__(self,
                 collection_name: str,
                 factors: int = cfg.factors,
                 learning_rate=cfg.learning_rate,
                 alpha=1.0,
                 regularization: float = cfg.regularization,
                 iterations: int = cfg.iterations,
                 threshold: int = 2,
                 train_months_duration: int = 5):

        super().__init__(collection_name,threshold=threshold,
                         train_months_duration=train_months_duration,
                         alpha=alpha)

        self.model = BayesianPersonalizedRanking(factors=factors,
                                                 learning_rate=learning_rate,
                                                 regularization=regularization,
                                                 iterations=iterations)
        self.fit()

    def fit(self):

        self.model.fit(self.train_iu)
        return self.model

    def recommend(self,
                  username: str,
                  k: Optional[int] = 5,
                  exclude_past: Optional[bool] = True):
        to_remove = None
        if exclude_past:
            to_remove = [self.dao.item2id(el) for el in
                         list(self.train_df.loc[self.train_df['users'] == username, 'items'])]

        user_id = self.dao.user2id(username)
        count_error = 0
        try:
            pre_ = self.model.recommend(userid=user_id,
                                    user_items=self.tot_ui,
                                    N=k,
                                    filter_already_liked_items=exclude_past,
                                    filter_items=to_remove)

        except IndexError as e:
            count_error += 1
            pre_ = []
            print(f'{e=} {user_id=} {count_error=}')

        recommended_items = [self.dao.id2item(idx=p[0]) for p in pre_]
            #print(recommended_items)
        return recommended_items

    def recommend_all_users(self,
                            k: Optional[int] = 10):
        out = {}
        print(self.ifp_test.n_users)
        for u in self.ifp_test.users:
            user_id = self.dao.user2id(username=u)
            out[u] = self.recommend(username=u, k=k)
        return out


class Try(BaseRecommender):
    pass


if __name__ == '__main__':
    a = BaseRecommender(collection_name="chiara_ferragni")
    print(f'{a.ifp_tot.n_users=}')
    print(f'{a.ifp_train.n_users=}')
    print(f'{a.ifp_test.n_users=}')
    x = ALS(collection_name="chiara_ferragni")
    #x.recommend(username="fef09ff875010a22648ae22913457431")
    print(len(x.recommend_all_users()))
