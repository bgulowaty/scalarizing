from .utils import top_n_indicies, create_voting_classifier
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, recall_score, precision_score
import numpy as np
from deslib.util import diversity
from itertools import combinations


def default_scoring_function(scorer, folds_iterator, classifiers, ensemble_size):
    calculated_test_accuracies = []

    for x_train, y_train, x_test, y_test, train_accuracies, test_accuracies, train_predictions_by_clf, test_predictions_by_clf in folds_iterator:
        scores_for_single_fold = []

        for clf_idx, clf in enumerate(classifiers):  # Calculate score for every clf for this fold
            score = scorer(y_train, train_predictions_by_clf[clf_idx])
            scores_for_single_fold.append(score)

        best_clf_indices_by_test_accuracy = top_n_indicies(test_accuracies, ensemble_size)
        best_clf_indices = top_n_indicies(scores_for_single_fold, ensemble_size)

        if ensemble_size == 1:
            accuracies = {
                'by_score': test_accuracies[best_clf_indices[0]],
                'by_accuracy': test_accuracies[best_clf_indices_by_test_accuracy[0]]
            }

        else:
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

    for x_train, y_train, x_test, y_test, train_accuracies, test_accuracies, train_predictions_by_clf, test_predictions_by_clf in folds_iterator:
        scores_for_single_fold = []

        for clf_idx, clf in enumerate(classifiers):  # Calculate score for every clf for this fold
            score = scorer(y_train, train_predictions_by_clf[clf_idx])
            scores_for_single_fold.append(score)

        best_clf_indices_by_test_accuracy = top_n_indicies(test_accuracies, ensemble_size)
        best_clf_indices = top_n_indicies(scores_for_single_fold, ensemble_size)

        if ensemble_size == 1:
            accuracies = {
                'by_score': test_accuracies[best_clf_indices[0]],
                'by_accuracy': test_accuracies[best_clf_indices_by_test_accuracy[0]]
            }

        else:
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



