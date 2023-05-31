import loguru
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from box import Box
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
import pytest

from scalarizing.scalarizing import FindingBestExpressionSingleDatasetProblem, FindingBestExpressionProblemMutation, \
    FindingBestExpressionProblemCrossover, FindingBestExpressionProblemSampling
from scalarizing.scoring_functions import diversity_metric_scoring_function, default_scoring_function
from scalarizing.utils import extract_classifiers_from_bagging


def read_dataset(path):
    data = pd.read_csv(path)
    x = data.drop('TARGET', axis=1).values
    y = data['TARGET'].values

    return {
        "x": x,
        "y": y
    }


@pytest.mark.parametrize(
    "scoring_function",
    [default_scoring_function, diversity_metric_scoring_function],
)
def test_smoke(scoring_function):
    train = read_dataset('breast-train-0-s1.csv')
    test = read_dataset('breast-test-0-s1.csv')
    dataset = Box({
        'train': train,
        'test': test
    })

    bagging = BaggingClassifier(base_estimator=Perceptron(), n_estimators=200, max_samples=0.3)
    bagging.fit(dataset.train.x, dataset.train.y)

    problem = FindingBestExpressionSingleDatasetProblem(dataset.train, extract_classifiers_from_bagging(bagging),
                                                        ensemble_size=10, scoring_function=scoring_function)
    result = minimize(problem,
                      GA(
                          pop_size=5,
                          verbose=True,
                          seed=42,
                          eliminate_duplicates=False,
                          mutation=FindingBestExpressionProblemMutation(),
                          crossover=FindingBestExpressionProblemCrossover(),
                          sampling=FindingBestExpressionProblemSampling()
                      ),
                      ("n_gen", 5),
                      verbose=True,
                      save_history=False,
                      seed=42)
