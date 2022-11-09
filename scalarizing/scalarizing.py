import random
from dataclasses import dataclass
from itertools import cycle
from typing import List

import numpy as np
from imblearn.metrics import geometric_mean_score
from loguru import logger
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sympy import symbols, parse_expr

from rules.utils.sympy_utils import get_all_possible_expression_addresses, modify_expression
from scalarizing.scoring_functions import default_scoring_function
from scalarizing.utils import np_cache

balanced_accuracy = symbols('balanced_accuracy')
f1 = symbols('f1')
accuracy = symbols('accuracy')
g_mean = symbols('g_mean')
recall = symbols('recall')
precision = symbols('precision')

all_symbols = [balanced_accuracy, f1, accuracy, g_mean, recall, precision]

random_symbol = lambda: random.choice(all_symbols)
random_weight = lambda: random.uniform(0, 1)
random_symbol_or_weight = lambda: random.choice([random_symbol, random_weight])()

possible_modifiers = [
    lambda expr: expr + random_symbol(),
    lambda expr: expr - random_symbol(),
    lambda expr: expr / random_symbol(),
    lambda expr: expr * random_symbol(),
    lambda expr: expr ** random_symbol(),
    lambda expr: None
]


def modify_random_part(expr):
    all_addresses = get_all_possible_expression_addresses(expr)
    selected_address = random.choice(all_addresses)
    selected_modifier = random.choice(possible_modifiers)
    return modify_expression(expr, selected_modifier, selected_address)


symbols_iter = cycle([*all_symbols])


def scorer_creator(the_expression, labels=None):
    def my_custom_loss_func(y_true, y_pred):
        subs = {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
            'g_mean': geometric_mean_score(y_true, y_pred, average='weighted', labels=labels),
            'recall': recall_score(y_true, y_pred, average='weighted', labels=labels),
            'f1': f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        }
        result = parse_expr(str(the_expression)).evalf(subs=subs)
        try:
            return float(result)
        except Exception as e:
            logger.error(e)
            logger.error(result)
            return 0

    return my_custom_loss_func


class FindingBestExpressionProblemSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            X[i, 0] = random_symbol()

        return X


class FindingBestExpressionProblemMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):

        # for each individual
        for i in range(len(X)):
            r = np.random.random()

            if r < 0.5:
                new_i = modify_random_part(X[i, 0])
                X[i, 0] = new_i

        return X


class FindingBestExpressionProblemCrossover(Crossover):
    def __init__(self):
        super().__init__(1, 1)

    def _do(self, problem, X, **kwargs):
        return X



@dataclass
class ScoringFunctionArguments:
    score: float
    y_true: List
    y_pred: List


class FindingBestExpressionSingleDatasetProblem(ElementwiseProblem):

    def __init__(self, dataset, classifiers, ensemble_size=3, splitter=StratifiedKFold(n_splits=3), scoring_function = default_scoring_function, *args, **kwargs):
        super().__init__(n_var=1,  # we minimize for one specific metric
                         n_obj=1,  # we treat the expression as single variable
                         n_constr=0,
                         *args,
                         **kwargs)


        self.scoring_function = scoring_function
        self.labels = np.unique(dataset.y)
        self.ensemble_size = ensemble_size
        self.dataset = dataset
        self.classifiers = np.copy(classifiers)
        self.train_idx = []
        self.test_idx = []
        self.predictions = []

        for clf in classifiers:
            clf.predict = np_cache(maxsize=None)(clf.predict)

        for train_idx, test_idx in splitter.split(dataset.x, dataset.y):
            self.train_idx.append(train_idx)
            self.test_idx.append(test_idx)

    def __str__(self):
        return f"""
            ensemble_size={self.ensemble_size}
            classifiers={self.classifiers}
            train_idx={self.train_idx}
        """


    def _evaluate(self, individual, out, *args, **kwargs):
        scorer = scorer_creator(individual[0], labels=self.labels)  # individual has just the expression

        out["F"] = self.scoring_function(scorer, self.folds_iterator(), self.classifiers, self.ensemble_size)

    def folds_iterator(self):
        for fold_idx, (train_idx, test_idx) in enumerate(zip(self.train_idx, self.test_idx)):
            yield self.dataset.x[train_idx], \
                  self.dataset.y[train_idx], \
                  self.dataset.x[test_idx], \
                  self.dataset.y[test_idx]