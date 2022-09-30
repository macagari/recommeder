# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import scipy.sparse as sp
#
# from typing import List
#
#
# # Code from:
# # https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
#
#
#
# def recommender_precision(predicted: List[list], actual: List[list]) -> int:
#     """
#     Computes the precision of each user's list of recommendations, and averages precision over all users.
#     ----------
#     actual : a list of lists
#         Actual items to be predicted
#         example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
#     predicted : a list of lists
#         Ordered predictions
#         example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
#     Returns:
#     -------
#         precision: int
#     """
#     def calc_precision(predicted_, actual_):
#         prec = [value for value in predicted_ if value in actual_]
#         prec = np.round(float(len(prec)) / float(len(predicted_)), 4)
#         return prec
#
#     precision = np.mean(list(map(calc_precision, predicted, actual)))
#     return precision
#
#
# # Code from:
# # https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
# def recommender_recall(predicted: List[list], actual: List[list]) -> int:
#     """
#     Computes the recall of each user's list of recommendations, and averages precision over all users.
#     ----------
#     actual : a list of lists
#         Actual items to be predicted
#         example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
#     predicted : a list of lists
#         Ordered predictions
#         example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
#     Returns:
#     -------
#         recall: int
#     """
#     def calc_recall(predicted_, actual_):
#         reca = [value for value in predicted_ if value in actual_]
#         reca = np.round(float(len(reca)) / (float(len(actual_)) + 0.00001), 4)
#         return reca
#
#     recall = np.mean(list(map(calc_recall, predicted, actual)))
#     return recall
#
#
# # Code from:
# # https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
# def personalization(predicted: List[list]) -> float:
#     """
#     Personalization measures recommendation similarity across users.
#     A high score indicates good personalization (user's lists of recommendations are different).
#     A low score indicates poor personalization (user's lists of recommendations are very similar).
#     A model is "personalizing" well if the set of recommendations for each user is different.
#     Parameters:
#     ----------
#     predicted : a list of lists
#         Ordered predictions
#         example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
#     Returns:
#     -------
#         The personalization score for all recommendations.
#     """
#
#     def make_rec_matrix(predicted_: List[list]) -> sp.csr_matrix:
#         df = pd.DataFrame(data=predicted_).reset_index().melt(
#             id_vars='index', value_name='item',
#         )
#         df = df[['index', 'item']].pivot(index='index', columns='item', values='item')
#         df = pd.notna(df)*1
#         rec_matrix = sp.csr_matrix(df.values)
#         return rec_matrix
#
#     # create matrix for recommendations
#     if type(predicted[0][0]) == str:
#         d = load_dict()['data2id']['items']
#
#         def item2data(sku: str):
#             return d[sku]
#
#         predicted = [list(map(item2data, el)) for el in predicted]
#     predicted = np.array(predicted, dtype=object)
#     rec_matrix_sparse = make_rec_matrix(predicted)
#
#     # calculate similarity for every user's recommendation list
#     similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)
#
#     # calculate average similarity
#     dim = similarity.shape[0]
#     personalization_ = (similarity.sum() - dim) / (dim * (dim - 1))
#     return 1 - personalization_
