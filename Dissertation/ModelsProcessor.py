class ModelsProcessor(object):
    def __init__(self
                 , seed = 999
                 , n_splits = 10):
        

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
        self.models.add_model(model = GridSearchCV(estimator=GaussianProcessClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Gaussian Process')
        self.models.add_model(model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Deicison Tree')
        self.models.add_model(model = GridSearchCV(estimator=RandomForestClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Random Forest')
        self.models.add_model(model = GridSearchCV(estimator=AdaBoostClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'AdaBoost')
        self.models.add_model(model = GridSearchCV(estimator=GaussianNB(), param_grid={}, cv=tscv), model_name = 'Naive Bayes')
        #self.models.add_model(model = GridSearchCV(estimator=QuadraticDiscriminantAnalysis(), param_grid={}, cv=tscv), model_name = 'QDA')


    def set_dataset(self
                    , no_of_intervals_per_day = 8
                    , no_of_steps = 1
                    , interpolation_method = 'linear'
                    , train_start = '2009-01-01'
                    , train_end = '2009-12-31'
                    , test_start = '2010-01-01'
                    , test_end = '2010-12-31'
                    , dimensions = 3):

        from DB import DB
        from Dataset import Dataset
        from Models import Models
        import matplotlib.pyplot as plt


        db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')

        sql = """

            select 
            [TradingDatetime]
            , [Open]
            , [High]
            , [Low]
            , [Close]
            , [Volume]
            FROM [DBHKUDissertation].[dbo].[TableStock]
            where Ticker in ('sh000300')
            and Interval in (30)

        """

        df = db.query_to_dataframe(sql_string = sql)

        self.dataset = Dataset(data = db.query_to_dataframe(sql_string = sql), no_of_intervals_per_day=no_of_intervals_per_day, no_of_steps=no_of_steps)
        self.dataset.visualize(columns=['Close'])
        self.dataset.interpolate(method = interpolation_method)
        self.dataset.derive_features()
        self.dataset.remove_na()
        #dataset.get_data().head()['ReturnDummy'].unique()
        self.dataset.data_splitting(train_start = train_start, train_end = train_end, test_start = test_start, test_end = test_end)
        self.dataset.avoid_look_ahead_bias()
        self.dataset.set_X(['MA10', 'MA20', 'MA30', 'MA40', 'MA50', 'BB10', 'BB20', 'BB30', 'RSI10', 'RSI20', 'RSI30', 'STOCHK10', 'STOCHK20', 'STOCHK30', 'STOCHKD10', 'STOCHKD20', 'STOCHKD30'])
        self.dataset.set_y(['ReturnDummy'])
        self.dataset.set_X_train()
        self.dataset.set_y_train()
        self.dataset.set_X_test()
        self.dataset.set_y_test()
        self.dataset.normalization()
        self.dataset.dimension_reduction(n_components = dimensions)



    def get_dataset(self):
        return self.dataset

    def get_models(self):
        return self.models

    
    def train_validate_test(self):
        from sklearn.utils import column_or_1d
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, precision_recall_fscore_support, classification_report
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve


        plt.figure(1)
        for model_name, model in self.models.get_models().items():
            print(model_name)
            #print(model)

            model.fit(self.dataset.get_X_train(), column_or_1d(self.dataset.get_y_train()))
            y_pred = model.predict(self.dataset.get_X_test())

            #print(accuracy_score(self.dataset.get_y_test(), y_pred))
            print(classification_report(self.dataset.get_y_test(), y_pred))

            #print(model.best_estimator_)
            #print(model.best_score_)
            #print(model.best_params_)
            #print(model.cv_results_)

            fpr, tpr, _ = roc_curve(self.dataset.get_y_test(), y_pred)



            plt.plot(fpr, tpr, label=model_name)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()


        


