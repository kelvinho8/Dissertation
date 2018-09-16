class Processor(object):
    def __init__(self
                 , seed = 999
                 , n_splits = 10):
        

        self.seed = seed
        self.n_splits = n_splits

        import sys
        sys.path.append(r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")
        #sys.path.append(r"C:\Users\Kelvi\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")

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

        #self.models.add_model(model = GridSearchCV(estimator=MLPClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Neural Net')
        #self.models.add_model(model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={}, cv=tscv), model_name = 'KNN')
        #self.models.add_model(model = GridSearchCV(estimator=SVC(kernel='linear', random_state=seed), param_grid={}, cv=tscv), model_name = 'Linear SVM')
        #self.models.add_model(model = GridSearchCV(estimator=SVC(kernel='rbf', random_state=seed), param_grid={}, cv=tscv), model_name = 'RBF SVM')
        #self.models.add_model(model = GridSearchCV(estimator=GaussianProcessClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Gaussian Process')
        #self.models.add_model(model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Decision Tree')
        #self.models.add_model(model = GridSearchCV(estimator=RandomForestClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Random Forest')
        #self.models.add_model(model = GridSearchCV(estimator=AdaBoostClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'AdaBoost')
        #self.models.add_model(model = GridSearchCV(estimator=GaussianNB(), param_grid={}, cv=tscv), model_name = 'Naive Bayes')
        ##self.models.add_model(model = GridSearchCV(estimator=QuadraticDiscriminantAnalysis(), param_grid={}, cv=tscv), model_name = 'QDA')

        self.models.add_model(model = MLPClassifier(random_state=seed), model_name = 'Neural Net')
        self.models.add_model(model = KNeighborsClassifier(), model_name = 'KNN')
        self.models.add_model(model = SVC(kernel='linear', random_state=seed), model_name = 'Linear SVM')
        self.models.add_model(model = SVC(kernel='rbf', random_state=seed), model_name = 'RBF SVM')
        self.models.add_model(model = GaussianProcessClassifier(random_state=seed), model_name = 'Gaussian Process')
        self.models.add_model(model = DecisionTreeClassifier(random_state=seed), model_name = 'Decision Tree')
        self.models.add_model(model = RandomForestClassifier(random_state=seed), model_name = 'Random Forest')
        self.models.add_model(model = AdaBoostClassifier(random_state=seed), model_name = 'AdaBoost')
        self.models.add_model(model = GaussianNB(), model_name = 'Naive Bayes')
        #self.models.add_model(model = QuadraticDiscriminantAnalysis(), model_name = 'QDA')



    def set_dataset(self
                    , interval = 30
                    , no_of_steps = 1
                    , window_size = 10
                    , interpolation_method = 'linear'
                    , train_start = '2009-01-01'
                    , train_end = '2009-12-31'
                    , valid_start = '2010-01-01'
                    , valid_end = '2010-12-31'
                    , test_start = '2011-01-01'
                    , test_end = '2011-12-31'
                    , dimensions = 3):

        from DB import DB
        from Dataset import Dataset
        from Models import Models
        import matplotlib.pyplot as plt

        self.interval = interval
        self.no_of_steps = no_of_steps
        self.window_size = window_size
        self.interpolation_method = interpolation_method
        self.train_start = train_start
        self.train_end = train_end

        self.valid_start = valid_start
        self.valid_end = valid_end

        self.test_start = test_start
        self.test_end = test_end
        self.dimensions = dimensions
        
        db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\MSSQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')
        #db = DB(driver = '{SQL Server}', server = 'LAPTOP-194NACED\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')

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
            and Interval in (""" + str(interval) + """)

        """

        #df = db.read_sql(sql_string = sql)


        self.dataset = Dataset(data = db.read_sql(sql_string = sql), no_of_intervals_per_day=240 / interval, no_of_steps=no_of_steps)
        #print(self.dataset.get_data().head())
        #self.dataset.visualize(columns=['Close'])
        print('created dataset object')
        self.dataset.interpolate(method = interpolation_method)
        #print(self.dataset.get_data().head())
        
        print('interpolated')
        self.dataset.derive_features(window_size)
        #print(self.dataset.get_data().to_string())
        

        print('derived features')
        self.dataset.remove_na()
        print(self.dataset.get_data().head())
        
        print('removed na')
        #print(self.dataset.get_data().head())
        #dataset.get_data().head()['ReturnDummy'].unique()
        self.dataset.data_splitting(train_start = train_start, train_end = train_end, valid_start = valid_start, valid_end = valid_end, test_start = test_start, test_end = test_end)
        #print(self.dataset.get_train().head())
        
        print('splitted data')
        self.dataset.avoid_look_ahead_bias()
        #print(self.dataset.get_train().head())
        
        print('removed look ahead')
        #self.dataset.set_X(['MA' + str(window_size), 'MA' + str(window_size * 2), 'MA' + str(window_size * 3), 'MA' + str(window_size * 4), 'MA' + str(window_size * 5), 'BB' + str(window_size), 'BB' + str(window_size * 2), 'BB' + str(window_size * 3), 'RSI' + str(window_size), 'RSI' + str(window_size * 2), 'RSI' + str(window_size * 3), 'STOCHK' + str(window_size), 'STOCHK' + str(window_size * 2), 'STOCHK' + str(window_size * 3), 'STOCHKD' + str(window_size), 'STOCHKD' + str(window_size * 2), 'STOCHKD' + str(window_size * 3)])
        self.dataset.set_X(['MA' + str(window_size), 'MA' + str(window_size * 2), 'MA' + str(window_size * 3), 'BB' + str(window_size), 'BB' + str(window_size * 2), 'BB' + str(window_size * 3), 'RSI' + str(window_size), 'RSI' + str(window_size * 2), 'RSI' + str(window_size * 3), 'STOCHK' + str(window_size), 'STOCHK' + str(window_size * 2), 'STOCHK' + str(window_size * 3), 'STOCHKD' + str(window_size), 'STOCHKD' + str(window_size * 2), 'STOCHKD' + str(window_size * 3)])
        
        #print(self.dataset.get_train().head())
        
        print('set X features')

        self.dataset.set_y(['ReturnDummy'])
        print('set Y feature')

        self.dataset.set_X_train()
        #print(self.dataset.get_X_train().head())
        print('set X_train')

        self.dataset.set_y_train()
        print('set y_train')

        self.dataset.set_X_valid()
        print('set X_valid')

        self.dataset.set_y_valid()
        print('set y_valid')

        self.dataset.set_X_test()
        print('set X_test')

        self.dataset.set_y_test()
        print('set y_test')

        self.dataset.normalization()
        print('normalization')

        self.dataset.dimension_reduction(n_components = dimensions)
        print('dimension reduction')

        self.dataset.print_train_test_period()


    def get_dataset(self):
        return self.dataset

    def get_models(self):
        return self.models

    
    def train_validate_test_models(self):
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



    def train_validate_test_model(self, model_name):
        from sklearn.utils import column_or_1d
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, precision_recall_fscore_support, classification_report

        print(model_name)
        model = self.models.get_model(model_name = model_name)
        model.fit(self.dataset.get_X_train(), column_or_1d(self.dataset.get_y_train()))

        y_train_pred = model.predict(self.dataset.get_X_train())
        y_valid_pred = model.predict(self.dataset.get_X_valid())
        y_test_pred = model.predict(self.dataset.get_X_test())
        
        print(confusion_matrix(self.dataset.get_y_train(), y_train_pred))
        print(confusion_matrix(self.dataset.get_y_valid(), y_valid_pred))
        print(confusion_matrix(self.dataset.get_y_test(), y_test_pred))

        train_tn, train_fp, train_fn, train_tp = confusion_matrix(self.dataset.get_y_train(), y_train_pred).ravel()
        valid_tn, valid_fp, valid_fn, valid_tp = confusion_matrix(self.dataset.get_y_valid(), y_valid_pred).ravel()
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(self.dataset.get_y_test(), y_test_pred).ravel()
        
        return accuracy_score(self.dataset.get_y_train(), y_train_pred), f1_score(self.dataset.get_y_train(), y_train_pred), precision_score(self.dataset.get_y_train(), y_train_pred), recall_score(self.dataset.get_y_train(), y_train_pred), train_tn, train_fp, train_fn, train_tp, accuracy_score(self.dataset.get_y_valid(), y_valid_pred), f1_score(self.dataset.get_y_valid(), y_valid_pred), precision_score(self.dataset.get_y_valid(), y_valid_pred), recall_score(self.dataset.get_y_valid(), y_valid_pred), valid_tn, valid_fp, valid_fn, valid_tp, accuracy_score(self.dataset.get_y_test(), y_test_pred), f1_score(self.dataset.get_y_test(), y_test_pred), precision_score(self.dataset.get_y_test(), y_test_pred), recall_score(self.dataset.get_y_test(), y_test_pred), test_tn, test_fp, test_fn, test_tp





    #def objective_function(self
    #                        , interval = 30
    #                        , no_of_steps = 1
    #                        , window_size = 10
    #                        , interpolation_method = 'linear'
    #                        , train_start = '2009-01-01'
    #                        , train_end = '2009-12-31'
    #                        , test_start = '2010-01-01'
    #                        , test_end = '2010-12-31'
    #                        , dimensions = 3
    #                        , model_name = 'RBF SVM'):
    #    self.set_dataset(interval = interval
    #                    , no_of_steps = no_of_steps
    #                    , window_size = window_size
    #                    , interpolation_method = interpolation_method
    #                    , train_start = train_start
    #                    , train_end = train_end
    #                    , test_start = test_start
    #                    , test_end = test_end
    #                    , dimensions = dimensions)


    #    return self.train_validate_test_model(model_name)


    def exhaustive_grid_search(self
                    , interval = 30
                    , min_no_of_steps = 10
                    , max_no_of_steps = 101
                    , no_of_steps_interval = 5
                    , min_window_size = 5
                    , max_window_size = 100
                    , window_size_interval = 5
                    , interpolation_method = 'linear'
                    , train_start = '2009-01-01'
                    , train_end = '2009-12-31'
                    , valid_start = '2010-01-01'
                    , valid_end = '2010-12-31'
                    , test_start = '2011-01-01'
                    , test_end = '2011-12-31'
                    , dimensions = 3
                    , model_name = 'RBF SVM'):

        import numpy as np
        import pandas as pd
        #import matplotlib.pyplot as plt
        #from mpl_toolkits.mplot3d import Axes3D
        #from matplotlib import cm

        self.model_name = model_name

        #print(min_no_of_steps)
        no_of_steps = np.concatenate([[1], np.arange(min_no_of_steps, max_no_of_steps + 1, no_of_steps_interval)])
        window_size = np.arange(min_window_size, max_window_size + 1, window_size_interval)

        #print(no_of_steps)
        #print(window_size)
        X, Y = np.meshgrid(no_of_steps, window_size)

        #print(X)
        #print(Y)
        #print(type(X))
        X_train_start_date = []
        X_train_end_date = []
        X_valid_start_date = []
        X_valid_end_date = []
        X_test_start_date = []
        X_test_end_date = []


        train_accuracy = np.zeros(X.shape)
        train_F1 = np.zeros(X.shape)
        train_precision = np.zeros(X.shape)
        train_recall = np.zeros(X.shape)
        
        train_true_false_ratio = np.zeros(X.shape)
        train_no_of_true = np.zeros(X.shape)
        train_no_of_false = np.zeros(X.shape)

        train_tn = np.zeros(X.shape)
        train_fp = np.zeros(X.shape)
        train_fn = np.zeros(X.shape)
        train_tp = np.zeros(X.shape)



        valid_accuracy = np.zeros(X.shape)
        valid_F1 = np.zeros(X.shape)
        valid_precision = np.zeros(X.shape)
        valid_recall = np.zeros(X.shape)
        
        valid_true_false_ratio = np.zeros(X.shape)
        valid_no_of_true = np.zeros(X.shape)
        valid_no_of_false = np.zeros(X.shape)

        valid_tn = np.zeros(X.shape)
        valid_fp = np.zeros(X.shape)
        valid_fn = np.zeros(X.shape)
        valid_tp = np.zeros(X.shape)


        test_accuracy = np.zeros(X.shape)
        test_F1 = np.zeros(X.shape)
        test_precision = np.zeros(X.shape)
        test_recall = np.zeros(X.shape)
        
        test_true_false_ratio = np.zeros(X.shape)
        test_no_of_true = np.zeros(X.shape)
        test_no_of_false = np.zeros(X.shape)

        test_tn = np.zeros(X.shape)
        test_fp = np.zeros(X.shape)
        test_fn = np.zeros(X.shape)
        test_tp = np.zeros(X.shape)

        seed = np.zeros(X.shape)
        n_splits = np.zeros(X.shape)


        rows = 0
        #print(Z)
        for step_row, window_row in zip(X, Y):
            

            columns = 0
            for step, window in zip(step_row, window_row):
                
                print(step)
                print(window)
                self.set_dataset(interval = interval
                , no_of_steps = step
                , window_size = window
                , interpolation_method = interpolation_method
                , train_start = train_start
                , train_end = train_end
                , valid_start = valid_start
                , valid_end = valid_end
                , test_start = test_start
                , test_end = test_end
                , dimensions = dimensions)
                
                train_accuracy[rows, columns], train_F1[rows, columns], train_precision[rows, columns], train_recall[rows, columns], train_tn[rows, columns], train_fp[rows, columns], train_fn[rows, columns], train_tp[rows, columns], valid_accuracy[rows, columns], valid_F1[rows, columns], valid_precision[rows, columns], valid_recall[rows, columns], valid_tn[rows, columns], valid_fp[rows, columns], valid_fn[rows, columns], valid_tp[rows, columns], test_accuracy[rows, columns], test_F1[rows, columns], test_precision[rows, columns], test_recall[rows, columns], test_tn[rows, columns], test_fp[rows, columns], test_fn[rows, columns], test_tp[rows, columns] = self.train_validate_test_model(model_name)
                
                train_no_of_true[rows, columns] = self.dataset.get_y_train_true()
                train_no_of_false[rows, columns] = self.dataset.get_y_train_false()
                train_true_false_ratio[rows, columns] = self.dataset.get_y_train_ratio()

                valid_no_of_true[rows, columns] = self.dataset.get_y_valid_true()
                valid_no_of_false[rows, columns] = self.dataset.get_y_valid_false()
                valid_true_false_ratio[rows, columns] = self.dataset.get_y_valid_ratio()

                test_no_of_true[rows, columns] = self.dataset.get_y_test_true()
                test_no_of_false[rows, columns] = self.dataset.get_y_test_false()
                test_true_false_ratio[rows, columns] = self.dataset.get_y_test_ratio()


                
                X_train_start_date.append(self.dataset.get_X_train_start_date().strftime("%Y-%m-%d %H:%M:%S"))
                X_train_end_date.append(self.dataset.get_X_train_end_date().strftime("%Y-%m-%d %H:%M:%S"))

                X_valid_start_date.append(self.dataset.get_X_valid_start_date().strftime("%Y-%m-%d %H:%M:%S"))
                X_valid_end_date.append(self.dataset.get_X_valid_end_date().strftime("%Y-%m-%d %H:%M:%S"))

                X_test_start_date.append(self.dataset.get_X_test_start_date().strftime("%Y-%m-%d %H:%M:%S"))
                X_test_end_date.append(self.dataset.get_X_test_end_date().strftime("%Y-%m-%d %H:%M:%S"))

                seed[rows, columns] = self.seed
                n_splits[rows, columns] = self.n_splits

                #Z[rows, columns] = self.objective_function(interval = interval
                #                    , no_of_steps = step
                #                    , window_size = window
                #                    , interpolation_method = interpolation_method
                #                    , train_start = train_start
                #                    , train_end = train_end
                #                    , test_start = test_start
                #                    , test_end = test_end
                #                    , dimensions = dimensions
                #                    , model_name = model_name)

                columns = columns + 1

            rows = rows + 1

        print(train_accuracy)
        print(X.reshape(X.size, 1)[:, 0])
        print(Y.reshape(Y.size, 1)[:, 0])
        print(train_accuracy.reshape(train_accuracy.size, 1)[:, 0])

        df = pd.DataFrame(data = {'interpolation_method':interpolation_method
                                  , 'train_start':train_start
                                  , 'train_end':train_end
                                  
                                  , 'valid_start':valid_start
                                  , 'valid_end':valid_end
                                  
                                  , 'test_start':test_start
                                  , 'test_end':test_end
                                  , 'dimensions':dimensions
                                  , 'model_name':model_name
                                  , 'no_of_steps':X.reshape(X.size, 1)[:, 0]
                                  , 'window_size':Y.reshape(Y.size, 1)[:, 0]

                                  , 'train_accuracy_score':train_accuracy.reshape(train_accuracy.size, 1)[:, 0]
                                  , 'train_F1_score':train_F1.reshape(train_F1.size, 1)[:, 0]
                                  , 'train_precision_score':train_precision.reshape(train_precision.size, 1)[:, 0]
                                  , 'train_recall_score':train_recall.reshape(train_recall.size, 1)[:, 0]
                                  , 'train_roc_tn':train_tn.reshape(train_tn.size, 1)[:, 0]
                                  , 'train_roc_fp':train_fp.reshape(train_fp.size, 1)[:, 0]
                                  , 'train_roc_fn':train_fn.reshape(train_fn.size, 1)[:, 0]
                                  , 'train_roc_tp':train_tp.reshape(train_tp.size, 1)[:, 0]
                                  , 'y_train_ratio':train_true_false_ratio.reshape(train_true_false_ratio.size, 1)[:, 0]
                                  , 'y_train_true':train_no_of_true.reshape(train_no_of_true.size, 1)[:, 0]
                                  , 'y_train_false':train_no_of_false.reshape(train_no_of_false.size, 1)[:, 0]

                                  , 'valid_accuracy_score':valid_accuracy.reshape(valid_accuracy.size, 1)[:, 0]
                                  , 'valid_F1_score':valid_F1.reshape(valid_F1.size, 1)[:, 0]
                                  , 'valid_precision_score':valid_precision.reshape(valid_precision.size, 1)[:, 0]
                                  , 'valid_recall_score':valid_recall.reshape(valid_recall.size, 1)[:, 0]
                                  , 'valid_roc_tn':valid_tn.reshape(valid_tn.size, 1)[:, 0]
                                  , 'valid_roc_fp':valid_fp.reshape(valid_fp.size, 1)[:, 0]
                                  , 'valid_roc_fn':valid_fn.reshape(valid_fn.size, 1)[:, 0]
                                  , 'valid_roc_tp':valid_tp.reshape(valid_tp.size, 1)[:, 0]
                                  , 'y_valid_ratio':valid_true_false_ratio.reshape(valid_true_false_ratio.size, 1)[:, 0]
                                  , 'y_valid_true':valid_no_of_true.reshape(valid_no_of_true.size, 1)[:, 0]
                                  , 'y_valid_false':valid_no_of_false.reshape(valid_no_of_false.size, 1)[:, 0]

                                  , 'test_accuracy_score':test_accuracy.reshape(test_accuracy.size, 1)[:, 0]
                                  , 'test_F1_score':test_F1.reshape(test_F1.size, 1)[:, 0]
                                  , 'test_precision_score':test_precision.reshape(test_precision.size, 1)[:, 0]
                                  , 'test_recall_score':test_recall.reshape(test_recall.size, 1)[:, 0]
                                  , 'test_roc_tn':test_tn.reshape(test_tn.size, 1)[:, 0]
                                  , 'test_roc_fp':test_fp.reshape(test_fp.size, 1)[:, 0]
                                  , 'test_roc_fn':test_fn.reshape(test_fn.size, 1)[:, 0]
                                  , 'test_roc_tp':test_tp.reshape(test_tp.size, 1)[:, 0]
                                  , 'y_test_ratio':test_true_false_ratio.reshape(test_true_false_ratio.size, 1)[:, 0]
                                  , 'y_test_true':test_no_of_true.reshape(test_no_of_true.size, 1)[:, 0]
                                  , 'y_test_false':test_no_of_false.reshape(test_no_of_false.size, 1)[:, 0]

                                  , 'x_train_start_date':X_train_start_date
                                  , 'x_train_end_date':X_train_end_date
                                  
                                  , 'x_valid_start_date':X_valid_start_date
                                  , 'x_valid_end_date':X_valid_end_date
                                  
                                  , 'x_test_start_date':X_test_start_date
                                  , 'x_test_end_date':X_test_end_date
                                  , 'seed':seed.reshape(seed.size, 1)[:, 0]
                                  , 'n_splits':n_splits.reshape(n_splits.size, 1)[:, 0]

                                  , 'interval':interval}
                          , index = np.arange(1, train_accuracy.size + 1))

        self.X = X
        self.Y = Y

        self.train_accuracy = train_accuracy
        self.train_F1 = train_F1
        self.train_precision = train_precision
        self.train_recall = train_recall
        self.train_tn = train_tn
        self.train_fp = train_fp
        self.train_fn = train_fn
        self.train_tp = train_tp

        self.valid_accuracy = valid_accuracy
        self.valid_F1 = valid_F1
        self.valid_precision = valid_precision
        self.valid_recall = valid_recall
        self.valid_tn = valid_tn
        self.valid_fp = valid_fp
        self.valid_fn = valid_fn
        self.valid_tp = valid_tp

        
        self.test_accuracy = test_accuracy
        self.test_F1 = test_F1
        self.test_precision = test_precision
        self.test_recall = test_recall
        self.test_tn = test_tn
        self.test_fp = test_fp
        self.test_fn = test_fn
        self.test_tp = test_tp

        self.grid_search_ouput = df

        return X, Y, train_accuracy, train_F1, train_precision, train_recall, train_tn, train_fp, train_fn, train_tp, valid_accuracy, valid_F1, valid_precision, valid_recall, valid_tn, valid_fp, valid_fn, valid_tp, test_accuracy, test_F1, test_precision, test_recall, test_tn, test_fp, test_fn, test_tp, df


    def to_sql(self, table_name, if_exists):
        from DB import DB
        
        db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\MSSQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')
        #db = DB(driver = '{SQL Server}', server = 'LAPTOP-194NACED\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')

        db.to_sql(df = self.grid_search_ouput, table_name = table_name, if_exists = if_exists)



    def delete_table(self, sql_string):
        from DB import DB
        
        db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\MSSQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')
        #db = DB(driver = '{SQL Server}', server = 'LAPTOP-194NACED\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')

        db.delete_table(sql_string)




    def plot_grid_search(self, is_saved, save_path):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        fig = plt.figure(figsize=(15, 9))
        #fig = plt.figure()

        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, self.valid_accuracy, rstride = 2, cstride = 2, cmap = cm.coolwarm, linewidth = 0.5, antialiased = True)
        ax.view_init(elev=75, azim=-50)
        ax.set_title('Steps vs Window Size Optimization\n, model = ' + str(self.model_name) + ', intervale = ' + str(self.interval) + ', dimensions = ' + str(self.dimensions) + '\n, train_start = ' + str(self.train_start) + ', train_end = ' + str(self.train_end) + ', valid_start = ' + str(self.valid_start) + ', valid_end = ' + str(self.valid_end) + ', test_start = ' + str(self.test_start) + ', test_end = ' + str(self.test_end))
        
        ax.set_xlabel('x = no_of_steps')
        ax.set_ylabel('y = window_size')
        ax.set_zlabel('f(x, y) = valid_accuracy_score')
        fig.colorbar(surf, shrink = 0.5, aspect = 5)
        plt.grid(True)

        if is_saved == 1:
            #filename = save_path + "\" + str(self.model_name) + "_" + str(self.interval) + '_' + str(self.dimensions) + '_' + str(self.train_start) + '_' + str(self.train_end) + '_' + str(self.test_start) + '_' + str(self.test_end) + '.png'
            filename = save_path + '\\' + str(self.model_name) + "_" + str(self.interval) + '_' + str(self.dimensions) + '_' + str(self.train_start) + '_' + str(self.train_end) + '_' + str(self.valid_start) + '_' + str(self.valid_end) + '_' + str(self.test_start) + '_' + str(self.test_end) + '.png'
            print(filename)
            plt.savefig(fname=filename)
        else:
            plt.show()


    def exhaustive_grid_search_models(self
                                        , interval = 30
                                        , min_no_of_steps = 5
                                        , max_no_of_steps = 301
                                        , no_of_steps_interval = 5
                                        , min_window_size = 5
                                        , max_window_size = 51
                                        , window_size_interval = 5
                                        , interpolation_method = 'linear'
                                        , train_start = '2009-01-01'
                                        , train_end = '2009-12-31'
                                        , valid_start = '2010-01-01'
                                        , valid_end = '2010-12-31'
                                        , test_start = '2011-01-01'
                                        , test_end = '2011-12-31'
                                        , dimensions = 3):


        for model_name, model in self.models.get_models().items():
            print(model_name)

            X, Y, train_accuracy, train_F1, train_precision, train_recall, train_tn, train_fp, train_fn, train_tp, valid_accuracy, valid_F1, valid_precision, valid_recall, valid_tn, valid_fp, valid_fn, valid_tp, test_accuracy, test_F1, test_precision, test_recall, test_tn, test_fp, test_fn, test_tp, grid_search_df = self.exhaustive_grid_search(interval = interval
                                                                                    , min_no_of_steps = min_no_of_steps
                                                                                    , max_no_of_steps = max_no_of_steps
                                                                                    , no_of_steps_interval = no_of_steps_interval
                                                                                    , min_window_size = min_window_size
                                                                                    , max_window_size = max_window_size
                                                                                    , window_size_interval = window_size_interval
                                                                                    , interpolation_method = interpolation_method
                                                                                    , train_start = train_start
                                                                                    , train_end = train_end
                                                                                    , valid_start = valid_start
                                                                                    , valid_end = valid_end
                                                                                    , test_start = test_start
                                                                                    , test_end = test_end
                                                                                    , dimensions = dimensions
                                                                                    , model_name = model_name)


            self.to_sql(table_name = 'GridSearchResult', if_exists = 'append')
            self.plot_grid_search(is_saved = 1, save_path = r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation\GridSearchImage")
            #self.plot_grid_search(is_saved = 1, save_path = r"C:\Users\Kelvi\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation\GridSearchImage")


    