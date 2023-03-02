# Example - MNIST
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import IPython

example="breast"

# Long run - 1 hour. MNIST
if example == "mnist":
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier()
    IPython.embed()
    automl.fit(X_train, y_train)
    IPython.embed()
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

# Two minute example on sklearn's breast cancer data.
if example == "breast":
    print("running breast")
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_classification_example_tmp',
        output_folder='/tmp/autosklearn_classification_example_out',
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')
    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
