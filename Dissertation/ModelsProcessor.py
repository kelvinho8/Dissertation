class ModelsProcessor(object):
    def __init__(self
                 , seed = 999
                 , n_splits = 10):
        

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

        self.models.add_model(model = GridSearchCV(estimator=MLPClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Neural Net')
        self.models.add_model(model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={}, cv=tscv), model_name = 'KNN')
        self.models.add_model(model = GridSearchCV(estimator=SVC(kernel='linear', random_state=seed), param_grid={}, cv=tscv), model_name = 'Linear SVM')
        self.models.add_model(model = GridSearchCV(estimator=SVC(kernel='rbf', random_state=seed), param_grid={}, cv=tscv), model_name = 'RBF SVM')
        self.models.add_model(model = GridSearchCV(estimator=GaussianProcessClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Gaussian Process')
        self.models.add_model(model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Decision Tree')
        self.models.add_model(model = GridSearchCV(estimator=RandomForestClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'Random Forest')
        self.models.add_model(model = GridSearchCV(estimator=AdaBoostClassifier(random_state=seed), param_grid={}, cv=tscv), model_name = 'AdaBoost')
        self.models.add_model(model = GridSearchCV(estimator=GaussianNB(), param_grid={}, cv=tscv), model_name = 'Naive Bayes')
        #self.models.add_model(model = GridSearchCV(estimator=QuadraticDiscriminantAnalysis(), param_grid={}, cv=tscv), model_name = 'QDA')


    def set_dataset(self
                    , interval = 30
                    , no_of_steps = 1
                    , window_size = 10
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

        self.interval = interval
        self.no_of_steps = no_of_steps
        self.window_size = window_size
        self.interpolation_method = interpolation_method
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.dimensions = dimensions
        
        db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')
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
        #self.dataset.visualize(columns=['Close'])
        self.dataset.interpolate(method = interpolation_method)
        self.dataset.derive_features(window_size)
        #self.dataset.get_data().head()
        self.dataset.remove_na()
        #dataset.get_data().head()['ReturnDummy'].unique()
        self.dataset.data_splitting(train_start = train_start, train_end = train_end, test_start = test_start, test_end = test_end)
        self.dataset.avoid_look_ahead_bias()
        self.dataset.set_X(['MA' + str(window_size), 'MA' + str(window_size * 2), 'MA' + str(window_size * 3), 'MA' + str(window_size * 4), 'MA' + str(window_size * 5), 'BB' + str(window_size), 'BB' + str(window_size * 2), 'BB' + str(window_size * 3), 'RSI' + str(window_size), 'RSI' + str(window_size * 2), 'RSI' + str(window_size * 3), 'STOCHK' + str(window_size), 'STOCHK' + str(window_size * 2), 'STOCHK' + str(window_size * 3), 'STOCHKD' + str(window_size), 'STOCHKD' + str(window_size * 2), 'STOCHKD' + str(window_size * 3)])
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
        y_pred = model.predict(self.dataset.get_X_test())

        return accuracy_score(self.dataset.get_y_test(), y_pred)



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
                    , test_start = '2010-01-01'
                    , test_end = '2010-12-31'
                    , dimensions = 3
                    , model_name = 'RBF SVM'):

        import numpy as np
        import pandas as pd
        #import matplotlib.pyplot as plt
        #from mpl_toolkits.mplot3d import Axes3D
        #from matplotlib import cm

        self.model_name = model_name

        #print(min_no_of_steps)
        no_of_steps = np.concatenate([[1], np.arange(min_no_of_steps, max_no_of_steps, no_of_steps_interval)])
        window_size = np.arange(min_window_size, max_window_size, window_size_interval)

        #print(no_of_steps)
        #print(window_size)
        X, Y = np.meshgrid(no_of_steps, window_size)

        #print(X)
        #print(Y)
        #print(type(X))

        Z = np.zeros(X.shape)
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
                , test_start = test_start
                , test_end = test_end
                , dimensions = dimensions)
                
                Z[rows, columns] = self.train_validate_test_model(model_name)

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

        print(Z)
        print(X.reshape(X.size, 1)[:, 0])
        print(Y.reshape(Y.size, 1)[:, 0])
        print(Z.reshape(Z.size, 1)[:, 0])

        df = pd.DataFrame(data = {'interpolation_method':interpolation_method
                                  , 'train_start':train_start
                                  , 'train_end':train_end
                                  , 'test_start':test_start
                                  , 'test_end':test_end
                                  , 'dimensions':dimensions
                                  , 'model_name':model_name
                                  , 'no_of_steps':X.reshape(X.size, 1)[:, 0]
                                  , 'window_size':Y.reshape(Y.size, 1)[:, 0]
                                  , 'accuracy_score':Z.reshape(Z.size, 1)[:, 0]
                                  , 'interval':interval}
                          , index = np.arange(1, Z.size + 1))

        self.X = X
        self.Y = Y
        self.Z = Z
        self.grid_search_ouput = df


        #if isPlotted == 1:
        #    fig = plt.figure(figsize=(9, 6))
        #    ax = Axes3D(fig)
        #    surf = ax.plot_surface(X, Y, Z, rstride = 2, cstride = 2, cmap = cm.coolwarm, linewidth = 0.5, antialiased = True)
        #    #ax.set_title('Steps vs Window Size Optimization\n, model = ' + str(model_name) + ', intervale = ' + str(interval) + ', dimensions = ' + str(dimensions) + ', train_start = ' + str(train_start) + ', train_end = ' + str(train_end) + ', test_start = ' + str(test_start) + ', test_end = ' + str(test_end))
        #    ax.set_title('Steps vs Window Size Optimization')
            
        #    ax.text(x = 0.5, y = 0.5, z = 0.5, s = 'model = ' + str(model_name) + ', intervale = ' + str(interval) + ', dimensions = ' + str(dimensions) + '\n, train_start = ' + str(train_start) + ', train_end = ' + str(train_end) + ', test_start = ' + str(test_start) + ', test_end = ' + str(test_end))
            
        #    ax.set_xlabel('x = no_of_steps')
        #    ax.set_ylabel('y = window_size')
        #    ax.set_zlabel('f(x, y) = accuracy_score')
        #    fig.colorbar(surf, shrink = 0.5, aspect = 5)
        #    plt.grid(True)
        #    plt.savefig(fname=r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation\GridSearchImage\sample.png")
        #    plt.show()

        return X, Y, Z, df


    def to_sql(self, table_name, if_exists):
        from DB import DB
        
        db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')
        #db = DB(driver = '{SQL Server}', server = 'LAPTOP-194NACED\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')

        db.to_sql(df = self.grid_search_ouput, table_name = table_name, if_exists = if_exists)



    def delete_table(self, sql_string):
        from DB import DB
        
        db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')
        #db = DB(driver = '{SQL Server}', server = 'LAPTOP-194NACED\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')

        db.delete_table(sql_string)




    def plot_grid_search(self, is_saved, save_path):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        fig = plt.figure(figsize=(15, 9))
        #fig = plt.figure()

        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, self.Z, rstride = 2, cstride = 2, cmap = cm.coolwarm, linewidth = 0.5, antialiased = True)
        ax.view_init(elev=75, azim=-50)
        ax.set_title('Steps vs Window Size Optimization\n, model = ' + str(self.model_name) + ', intervale = ' + str(self.interval) + ', dimensions = ' + str(self.dimensions) + '\n, train_start = ' + str(self.train_start) + ', train_end = ' + str(self.train_end) + ', test_start = ' + str(self.test_start) + ', test_end = ' + str(self.test_end))
        
        ax.set_xlabel('x = no_of_steps')
        ax.set_ylabel('y = window_size')
        ax.set_zlabel('f(x, y) = accuracy_score')
        fig.colorbar(surf, shrink = 0.5, aspect = 5)
        plt.grid(True)

        if is_saved == 1:
            #filename = save_path + "\" + str(self.model_name) + "_" + str(self.interval) + '_' + str(self.dimensions) + '_' + str(self.train_start) + '_' + str(self.train_end) + '_' + str(self.test_start) + '_' + str(self.test_end) + '.png'
            filename = save_path + '\\' + str(self.model_name) + "_" + str(self.interval) + '_' + str(self.dimensions) + '_' + str(self.train_start) + '_' + str(self.train_end) + '_' + str(self.test_start) + '_' + str(self.test_end) + '.png'
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
                                        , test_start = '2010-01-01'
                                        , test_end = '2010-12-31'
                                        , dimensions = 3):


        for model_name, model in self.models.get_models().items():
            print(model_name)

            X, Y, Z, grid_search_df = self.exhaustive_grid_search(interval = interval
                                                , min_no_of_steps = min_no_of_steps
                                                , max_no_of_steps = max_no_of_steps
                                                , no_of_steps_interval = no_of_steps_interval
                                                , min_window_size = min_window_size
                                                , max_window_size = max_window_size
                                                , window_size_interval = window_size_interval
                                                , interpolation_method = interpolation_method
                                                , train_start = train_start
                                                , train_end = train_end
                                                , test_start = test_start
                                                , test_end = test_end
                                                , dimensions = dimensions
                                                , model_name = model_name)


            self.to_sql(table_name = 'GridSearchResult', if_exists = 'append')
            self.plot_grid_search(is_saved = 1, save_path = r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation\GridSearchImage")
            count = count + 1
