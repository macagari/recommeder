from typing import List, Optional

import numpy as np
import pandas as pd

# from metrics.recommender_metrics import recommender_precision, recommender_recall  # , personalization

# from model.similarityrecommender import SimilarityRecommender
from dao.general_dao import Dao
from recommenders.collaborative_filtering import ALS, LMF, BPR, BaseRecommender
from recommenders.similarity_recommender import SimilarityRecommender


class HybridRecommender:
    def __init__(self,
                 collection_name: str,
                 # cnn_recommenders: List[str],
                 # cnn_quotas: List[float],
                 cf_recommenders: List[str],
                 cf_quotas: List[float],
                 train_month_duration: Optional = 3,
                 threshold: Optional[int] = 2):

        self.collection_name = collection_name
        dao = Dao(collection_name=self.collection_name)
        self.users_list = list(set(dao.dict_users_items['users'].values()))
        quotas = cf_quotas  # quotas just for CF recommender
        # quotas = cnn_quotas + cf_quotas
        recommenders_list = cf_recommenders
        # recommenders = cnn_recommenders + cf_recommenders
        assert sum(quotas) == 1.0, 'quotas float must sum to one'
        assert len(recommenders_list) == len(quotas), \
            'mismatch: |recommenders|={}!={}=|quotas|'.format(len(recommenders_list), len(quotas))
        self.recommenders = {}
        self.quotas = quotas

        for i, modelname in enumerate(recommenders_list):
            # if i in range(len(cnn_recommenders)):
            #     self.recommenders[i] = SimilarityRecommender(modname=modelname,
            #                                                  train_month_duration=train_month_duration,
            #
            #                                                   threshold=threshold)
            # TODO: create new list instead of replacing values
            if modelname == "ALS":

                self.recommenders[modelname] = ALS(self.collection_name, threshold=threshold,
                                                   train_months_duration=train_month_duration)
            elif modelname == "LMF":
                self.recommenders[modelname] = LMF(self.collection_name, threshold=threshold,
                                                   train_months_duration=train_month_duration)
            elif modelname == "BPR":
                self.recommenders[modelname] = BPR(self.collection_name, threshold=threshold,
                                                   train_months_duration=train_month_duration)

        # model = self.recommenders[0]
        # if isinstance(model, SimilarityRecommender):
        #     self.user_list = model.train_users
        # else:
        #     self.user_list = model.ifp_test.users
        # self.test_df = model.test_df

    def recommend(self,
                  user: str,
                  k: Optional[int] = 10):
        contrib = [int(q * k) for q in self.quotas]
        while sum(contrib) < k:
            contrib[np.argmax(self.quotas)] += 1

        recommendations = []
        for recommender, rec_quota in zip(self.recommenders.values(), contrib):
            partial_recommendations = recommender.recommend(username=user, k=rec_quota)
            recommendations.extend(partial_recommendations)

        return recommendations

    def recommend_all_users(self,
                            k: Optional[int] = 20
                            ):

        recommendations = {}
        rec_total = {}
        for recommender in self.recommenders:
            print(f'{recommender=}')
            recommendations[recommender] = self.recommenders[recommender].recommend_all_users(k=k)
        # Users in all the recommenders method are the same.
        for recommender in self.recommenders:
            for user in recommendations[recommender]:
                if user not in rec_total:
                    rec_total[user] = recommendations[recommender][user]
                else:
                    rec_total[user] += recommendations[recommender][user]

        return {key:set(rec_total[key]) for key in rec_total}

    # def assess_model(self,
    #                  dict_recommendations: Optional[dict] = None,
    #                  k: Optional[int] = 10):
    #     if dict_recommendations is None:
    #         dict_recommendations = self.recommend_all_users(k=k)
    #
    #     assert len(next(iter(dict_recommendations.values()))) == k, 'mismatch for evaluation!'
    #
    #     user_metrics = []
    #     tot_recommendations_len = 0
    #     tot_relevants_len = 0
    #     list_predicted = []
    #     list_actual = []
    #     for user in self.user_list:
    #         recommended = set(dict_recommendations[user])
    #         tot_recommendations_len += len(recommended)
    #         relevants = set(list(self.test_df.loc[self.test_df[COLUMNS().user] == user, COLUMNS().item]))
    #         tot_relevants_len += len(relevants)
    #         list_actual.append(list(relevants))
    #         list_predicted.append(list(recommended))
    #         numerator = recommended & relevants
    #         flag_user = 0
    #         if len(numerator) >= 1:
    #             flag_user = 1
    #         user_metrics.append([user, len(numerator), flag_user])
    #     metrics_df = pd.DataFrame(user_metrics, columns=['user', 'catch', 'flag_user'])
    #     precision = round(recommender_precision(list_predicted, list_actual), 3)
    #     recall = round(recommender_recall(list_predicted, list_actual), 3)
    #     # personalization_score = round(personalization(list_predicted), 3)
    #     perc_satisfied_users = round(sum(metrics_df['flag_user']) / metrics_df.shape[0], 3)
    #
    #     print('Assesment for Hybrid model: with quotas: {}'.format(self.quotas))
    #     print('\tPrecision@{}: {}'.format(k, precision))
    #     print('\tRecall@{}: {}'.format(k, recall))
    #     # print('\tPersonalization score: {}'.format(personalization_score))
    #     print('\t% satisfied users: {}'.format(perc_satisfied_users))
    #


if __name__ == '__main__':
    # cnn_list = ["resnet18"]
    # cnn_q = [0.5]
    cf_list = ["ALS", "LMF", "BPR"]
    cf_q = [0.3, 0.5, 0.2]
    hr = HybridRecommender(collection_name='chiara_ferragni',
                           cf_recommenders=cf_list,
                           cf_quotas=cf_q)

    # hr.assess_model(k=100)
    # print(hr.recommend(user="dabc0a10c4de6d38af90ac996809740", k=5))
    print(hr.recommend_all_users(k=10))
