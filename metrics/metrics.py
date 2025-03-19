import numpy as np
from itertools import combinations


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def precision_at_k(recommended, true_positives, k):
        return len(set(recommended[:k]) & set(true_positives)) / k

    @staticmethod
    def recall_at_k(recommended, true_positives, k):
        return len(set(recommended[:k]) & set(true_positives)) / len(true_positives)

    @staticmethod
    def f1_score_at_k(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    def reciprocal_rank(recommended, true_positives):
        for i, item in enumerate(recommended, start=1):
            if item in true_positives:
                return 1 / i
        return 0

    @classmethod
    def average_precision(cls, recommended, true_positives):
        precisions = [cls.precision_at_k(recommended, true_positives, k + 1) for k, item in enumerate(recommended) if
                      item in true_positives]
        return np.mean(precisions) if precisions else 0

    @staticmethod
    def hit_rate(recommended, true_positives, k):
        return 1 if len(set(recommended[:k]) & set(true_positives)) > 0 else 0

    @staticmethod
    def dcg_at_k(recommended, true_positives, k):
        dcg = sum(
            [1 / np.log2(i + 2) if recommended[i] in true_positives else 0 for i in range(min(len(recommended), k))])
        return dcg

    @staticmethod
    def idcg_at_k(true_positives, k):
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_positives), k))])
        return idcg if idcg > 0 else 1

    @classmethod
    def ndcg_at_k(cls, recommended, true_positives, k):
        return cls.dcg_at_k(recommended, true_positives, k) / cls.idcg_at_k(true_positives, k)

    @staticmethod
    def coverage(recommended_items, all_possible_items):
        return len(set(recommended_items)) / len(set(all_possible_items))

    @staticmethod
    def percent_intersect(set1, set2, k):
        intersection = len(set(set1[:k]).intersection(set(set2[:k])))
        return 1 - (intersection / k)

    @classmethod
    def personalization(cls, recommended, k):
        distances = []
        for (items1, items2) in combinations(recommended, 2):
            distance = cls.percent_intersect(items1, items2, k)
            distances.append(distance)
        return sum(distances) / len(distances) if distances else 0

    @classmethod
    def evaluate_by_row(cls, recommended, true_positives, k):
        df_eval = recommended.merge(true_positives, on='user_id', how='inner')

        df_eval[f'precision@{k}'] = df_eval.apply(
            lambda row: cls.precision_at_k(row['recommended'], row['true_positives'], k),
            axis=1
        )
        df_eval[f'recall@{k}'] = df_eval.apply(
            lambda row: cls.recall_at_k(row['recommended'], row['true_positives'], k),
            axis=1
        )
        df_eval[f'f1@{k}'] = df_eval.apply(
            lambda row: cls.f1_score_at_k(row[f'precision@{k}'], row[f'recall@{k}']),
            axis=1
        )
        df_eval[f'MRR@{k}'] = df_eval.apply(
            lambda row: cls.reciprocal_rank(row['recommended'], row['true_positives']),
            axis=1
        )
        df_eval[f'MAP@{k}'] = df_eval.apply(
            lambda row: cls.average_precision(row['recommended'], row['true_positives']),
            axis=1
        )
        df_eval[f'HitRate@{k}'] = df_eval.apply(
            lambda row: cls.hit_rate(row['recommended'], row['true_positives'], k),
            axis=1
        )
        df_eval[f'NDCG@{k}'] = df_eval.apply(
            lambda row: cls.ndcg_at_k(row['recommended'], row['true_positives'], k),
            axis=1
        )
        return df_eval

    @classmethod
    def evaluate(cls, recommended, true_positives, k):
        df_eval = cls.evaluate_by_row(recommended, true_positives, k)

        # Collect all recommended and true positive items
        all_recommended_items = set(item for sublist in df_eval['recommended'] for item in sublist)
        all_possible_items = set(item for sublist in df_eval['recommended'] for item in sublist) | \
                             set(item for sublist in df_eval['true_positives'] for item in sublist)

        # Compute coverage
        coverage = cls.coverage(all_recommended_items, all_possible_items)

        # Compute personalization
        personalization = cls.personalization(df_eval['recommended'], k)

        metrics = df_eval[
            [f'precision@{k}', f'recall@{k}', f'f1@{k}', f'MRR@{k}', f'MAP@{k}', f'HitRate@{k}', f'NDCG@{k}']
        ].mean()
        metrics[f'coverage@{k}'] = coverage
        metrics[f'personalization@{k}'] = personalization
        return metrics
