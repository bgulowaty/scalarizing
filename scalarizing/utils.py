from copy import deepcopy

import ray as ray


class predict_wrapper(object):
    def __init__(self, predict_func, labels):
        self.predict_func = predict_func
        self.labels = labels

    def __call__(self, *args, **kwargs):
        return self.labels[self.predict_func(*args, **kwargs)]

def raise_not_implemented():
    raise NotImplemented("Predict proba is not supported")
def extract_classifiers_from_bagging(bagging):

    extracted = []
    for classifier in bagging.estimators_:
        cloned_classifier = deepcopy(classifier)
        cloned_classifier.predict = predict_wrapper(cloned_classifier.predict, bagging.classes_)
        cloned_classifier.predict_proba = raise_not_implemented

        extracted.append(cloned_classifier)

    return extracted


@ray.remote
def execute_in_ray(f, x):
    return f(x)

class RayParallelization:

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, f, X):
        results = [execute_in_ray.remote(f, x) for x in X]

        return ray.get(results)


    def __getstate__(self):
        state = self.__dict__.copy()
        return state


class ExecutorParallelization:

    def __init__(self, executor) -> None:
        super().__init__()
        self.executor = executor

    def __call__(self, f, X):
        jobs = [self.executor.submit(f, x) for x in X]
        return [job.result() for job in jobs]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("executor", None) # is not serializable
        return state