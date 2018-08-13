class ModelsProcessor(object):
    def __init__(self, seed = 999, n_splits = 10):
        import sys
        sys.path.append(r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")
        from Models import Models
        
        self.models = Models()
        from sklearn.model_selection import GridSearchCV
        
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits)

        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        self.models.add_model(model = GridSearchCV(estimator=MLPClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Neural Net')
        self.models.add_model(model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={}, cv=tscv), model_name = 'KNN')
        self.models.add_model(model = GridSearchCV(estimator=SVC(kernel='linear', random_state=seed), param_grid={}, cv=tscv), model_name = 'Linear SVM')
        self.models.add_model(model = GridSearchCV(estimator=SVC(kernel='rbf', random_state=seed), param_grid={}, cv=tscv), model_name = 'RBF SVM')
        self.models.add_model(model = GridSearchCV(estimator=GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=seed), param_grid={}, cv=tscv), model_name = 'Gaussian Process')
        self.models.add_model(model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Deicison Tree')
        self.models.add_model(model = GridSearchCV(estimator=RandomForestClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Random Forest')
        self.models.add_model(model = GridSearchCV(estimator=AdaBoostClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'AdaBoost')
        self.models.add_model(model = GridSearchCV(estimator=GaussianNB(), param_grid={}, cv=tscv), model_name = 'Naive Bayes')
        self.models.add_model(model = GridSearchCV(estimator=QuadraticDiscriminantAnalysis(), param_grid={}, cv=tscv), model_name = 'QDA')


    def get_models(self):
        return self.models

    
    def train_validate_test(self, X_train, y_train, X_test, y_test):
        from sklearn.utils import column_or_1d
        for model_name, model in self.models.get_models().items():
            print(model_name)
            print(model)

            model.fit(X_train, column_or_1d(y_train))
            y_pred = model.predict(X_test)

            print(accuracy_score(y_test, y_pred))
            print(model.best_estimator_)
            print(model.best_score_)
            print(model.best_params_)
            #print(model.cv_results_)
        


