from itertools import combinations

import numpy as np
from deslib.util import diversity
from joblib import delayed, Parallel
from sklearn.metrics import accuracy_score

from .utils import top_n_indicies, create_voting_classifier


def default_scoring_function(scorer, folds_iterator, classifiers, ensemble_size):
    calculated_test_accuracies = []

    for x_train, y_train, x_test, y_test in folds_iterator:

        test_accuracies = [accuracy_score(clf.predict(x_test), y_test) for clf in classifiers]

        scores_for_single_fold = Parallel(n_jobs=8, backend='threading')(
            delayed(lambda clf: scorer(y_train, clf.predict(x_train)))(clf) for clf in classifiers
        )

        best_clf_indices_by_test_accuracy = top_n_indicies(test_accuracies, ensemble_size)
        best_clf_indices = top_n_indicies(scores_for_single_fold, ensemble_size)


        voting_clf = create_voting_classifier(classifiers[best_clf_indices], x_train, y_train)
        voting_clf_by_test_acc = create_voting_classifier(classifiers[best_clf_indices_by_test_accuracy], x_train, y_train)

        accuracies = {
            'by_score': accuracy_score(y_test, voting_clf.predict(x_test)),
            'by_accuracy': accuracy_score(y_test, voting_clf_by_test_acc.predict(x_test))
        }

        calculated_test_accuracies.append(accuracies)


    function_values = {
        'by_accuracy': np.average([1 - accuracy['by_accuracy'] for accuracy in calculated_test_accuracies]),
        'by_score': np.average([1 - accuracy['by_score'] for accuracy in calculated_test_accuracies])
    }

    if function_values['by_accuracy'] > function_values['by_score']:
        return function_values['by_accuracy']
    else:
        return function_values['by_score']


def diversity_metric_scoring_function(scorer, folds_iterator, classifiers, ensemble_size):
    calculated_test_accuracies = []

    for x_train, y_train, x_test, y_test in folds_iterator:
        scores_for_single_fold = []

        test_accuracies = [accuracy_score(clf.predict(x_test), y_test) for clf in classifiers]


        for clf_idx, clf in enumerate(classifiers):  # Calculate score for every clf for this fold
            score = scorer(y_train, clf.predict(x_train))
            scores_for_single_fold.append(score)

        best_clf_indices_by_test_accuracy = top_n_indicies(test_accuracies, ensemble_size)
        best_clf_indices = top_n_indicies(scores_for_single_fold, ensemble_size)

        classifiers_selected_by_score = classifiers[best_clf_indices]

        diversity_of_ensemble = np.average(
            [diversity.Q_statistic(y_train, clf1.predict(x_train), clf2.predict(x_train))
             for (clf1, clf2) in combinations(classifiers_selected_by_score, 2)]
        )

        voting_clf = create_voting_classifier(classifiers_selected_by_score, x_train, y_train)
        voting_clf_by_test_acc = create_voting_classifier(classifiers[best_clf_indices_by_test_accuracy], x_train, y_train)

        scores = {
            'by_score': accuracy_score(y_test, voting_clf.predict(x_test)),
            'by_accuracy': accuracy_score(y_test, voting_clf_by_test_acc.predict(x_test)),
            'diversity': diversity_of_ensemble
        }

        calculated_test_accuracies.append(scores)

    function_values = {
        'by_accuracy': np.average([1 - accuracy['by_accuracy'] for accuracy in calculated_test_accuracies]),
        'by_score': np.average([1 - accuracy['by_score'] for accuracy in calculated_test_accuracies]),
        'diversity': np.average([accuracy['diversity'] for accuracy in calculated_test_accuracies])
    }

    return np.mean(function_values['diversity'] + function_values['by_accuracy'])
