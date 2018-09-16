

#$reset
#$cls


#import sys
#sys.path.append(r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")
##sys.path.append(r"C:\Users\Kelvi\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")


#from DB import DB
#from Dataset import Dataset
#from Models import Models
#import matplotlib.pyplot as plt



#db = DB(driver = '{SQL Server}', server = 'ENVY15-NOTEBOOK\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')
##db = DB(driver = '{SQL Server}', server = 'LAPTOP-194NACED\SQL2017', database = 'DBHKUDissertation', username = 'sa', password = 'sa.2017')


#interval = 30

#sql = """

#    select 
#    [TradingDatetime]
#    , [Open]
#    , [High]
#    , [Low]
#    , [Close]
#    , [Volume]
#    FROM [DBHKUDissertation].[dbo].[TableStock]
#    where Ticker in ('sh000300')
#    and Interval in (""" + str(interval) + """)

#"""

#df = db.query_to_dataframe(sql_string = sql)

#df.head(16)


#dataset = Dataset(data = db.query_to_dataframe(sql_string = sql), no_of_intervals_per_day = 8, no_of_steps = 1)

##dataset.get_data()[1:-1]

#dataset.interpolate(method = 'linear')


#dataset.get_data().head()

#dataset.get_return_dummy()

##dataset.get_data()[['Close', 'Volume']].iloc[:]


##dataset.derive_indicators()

##dataset.get_data().tail()

##dataset.visualize(columns=['Close', 'MA50', 'MA200'])

##dataset.talib2df(dataset.STOCH(window_size = 40)).plot()
##dataset.talib2df(dataset.STOCH(100, 100, 10)).plot()
##dataset.get_data()['STOCHKD30'].plot(secondary_y=True)

##plt.show()


#dataset.derive_features()



##dataset.get_data()

##upperband, middleband, lowerband = dataset.BB(10)

##close = dataset.get_data()['Close'].values

##type(close)


##dataset.talib2df(dataset.BB(30)).plot()
##dataset.get_data()['Close'].plot(secondary_y=True)

##plt.show()

##tran_bb = dataset.tran_BB(10)
##tran_bb[-1]


##for c, u, l, bb in zip(close[-100:-1], upperband[-100:-1], lowerband[-100:-1], tran_bb[-100:-1]):
##    print(c, u, l, bb)




##for c, u, l, bb in zip(close[1:100], upperband[1:100], lowerband[1:100], tran_bb[1:100]):
##    print(c, u, l, bb)


##type(tran_bb)

##dataset.talib2df(dataset.STOCH(100)).plot()
##dataset.get_data()['STOCHKD30'].plot(secondary_y=True)

##plt.show()


#dataset.remove_na()

##dataset.get_data().tail()

#dataset.get_data().head()['ReturnDummy'].unique()

##dataset.get_data().head()

#dataset.data_splitting(train_start = '2009-01-01', train_end = '2009-12-31', test_start = '2010-01-01', test_end = '2010-12-31')

##dataset.get_train()

##dataset.get_test()

#dataset.avoid_look_ahead_bias()

#dataset.get_train().tail()

#dataset.set_X(['MA10', 'MA20', 'MA30', 'MA40', 'MA50', 'BB10', 'BB20', 'BB30', 'RSI10', 'RSI20', 'RSI30', 'STOCHK10', 'STOCHK20', 'STOCHK30', 'STOCHKD10', 'STOCHKD20', 'STOCHKD30'])
#dataset.get_X()

#dataset.set_y(['ReturnDummy'])
#dataset.get_y()

#dataset.set_X_train()
#dataset.set_y_train()

#dataset.set_X_test()
#dataset.set_y_test()


#dataset.get_X_train().head()
#dataset.get_y_train().head()

#dataset.get_X_test().head()
#dataset.get_y_test().head()

#dataset.normalization()


#dataset.get_X_train().head()
#dataset.get_X_test().head()


#dataset.dimension_reduction(n_components = 3)


#dataset.get_X_train().head()
#dataset.get_X_test().head()

#dataset.get_y_train().head()

#from ModelsProcessor import ModelsProcessor

#seed = 999
#n_splits = 5
#models_processor = ModelsProcessor(seed = seed, n_splits = n_splits)

#models_processor.get_models().get_models()

#models_processor.train_validate_test(X_train=dataset.get_X_train(), y_train=dataset.get_y_train(), X_test=dataset.get_X_test(), y_test=dataset.get_y_test())


#====================================================================

#models_processor.plot_optimization(interval = 30
#                                    , min_no_of_steps = 1
#                                    , max_no_of_steps = 301
#                                    , no_of_steps_interval = 10
#                                    , min_window_size = 5
#                                    , max_window_size = 31
#                                    , window_size_interval = 5
#                                    , interpolation_method = 'linear'
#                                    , train_start = '2009-01-01'
#                                    , train_end = '2009-12-31'
#                                    , test_start = '2010-01-01'
#                                    , test_end = '2010-12-31'
#                                    , dimensions = 3
#                                    , model_name = 'Decision Tree')



#a = [1, 2, 3, 4]

#a

#import numpy as np
#nda = np.array(a)
#nda.reshape(nda, shape = (2, 2))

#nda = np.zeros(shape=(2, 2))
#print(nda)


#nda[0, 0]

#a = np.arange(1, 101, 1).T

#print(a)




$reset
$cls


import sys
sys.path.append(r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")
#sys.path.append(r"C:\Users\Kelvi\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")



from ModelsProcessor import ModelsProcessor


models_processor = ModelsProcessor(seed = 999, n_splits = 5)

models_processor.set_dataset(interval = 15
                    , no_of_steps = 300
                    , window_size = 100
                    , interpolation_method = 'linear'
                    , train_start = '2008-01-01'
                    , train_end = '2008-12-31'
                    , test_start = '2009-01-01'
                    , test_end = '2009-12-31'
                    , dimensions = 3)

#models_processor.get_dataset().get_X_train().head()
#models_processor.get_dataset().get_X_train().tail()


#models_processor.get_dataset().get_X_train().index[0]
#models_processor.get_dataset().get_X_train().index[-1]

#models_processor.get_dataset().get_X_test().index[0]
#models_processor.get_dataset().get_X_test().index[-1]


#models_processor.get_dataset.visualize(columns=['Close'])

#models_processor.get_dataset().get_data().head()


models_processor.get_models().get_models()

#models_processor.train_validate_test(X_train=dataset.get_X_train(), y_train=dataset.get_y_train(), X_test=dataset.get_X_test(), y_test=dataset.get_y_test())

#models_processor.train_validate_test_models()
models_processor.train_validate_test_model(model_name = 'KNN')

models_processor.get_models().get_models().keys()


X, Y, accuracy, F1, precision, recall, tn, fp, fn, tp, grid_search_df = models_processor.exhaustive_grid_search(interval = 15
                                                                        , min_no_of_steps = 5
                                                                        , max_no_of_steps = 30
                                                                        , no_of_steps_interval = 5
                                                                        , min_window_size = 5
                                                                        , max_window_size = 10
                                                                        , window_size_interval = 5
                                                                        , interpolation_method = 'linear'
                                                                        , train_start = '2008-01-01'
                                                                        , train_end = '2008-12-31'
                                                                        , test_start = '2009-01-01'
                                                                        , test_end = '2009-12-31'
                                                                        , dimensions = 3
                                                                        , model_name = 'RBF SVM')

models_processor.delete_table(sql_string = 'delete [DBHKUDissertation].[dbo].[GridSearchResult]')


models_processor.to_sql(table_name = 'GridSearchResult', if_exists = 'replace')
#models_processor.plot_grid_search(is_saved = 0, save_path = r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation\GridSearchImage")
models_processor.plot_grid_search(is_saved = 0, save_path = r"C:\Users\Kelvi\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation\GridSearchImage")

models_processor.delete_table(sql_string = 'delete [DBHKUDissertation].[dbo].[GridSearchResult]')


from dateutil.relativedelta import relativedelta
import datetime

#tempdate = datetime.datetime.strptime('2007-01-01', '%Y-%m-%d').date()

#(tempdate + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")

#tempdate.strftime("%Y-%m-%d %H:%M:%S")

#from joblib import Parallel, delayed
#import multiprocessing
#num_cores = multiprocessing.cpu_count()


#incremental_years = range(1, 11)

#for incremental_year in incremental_years:
#    print(incremental_year)



#def run_yearly_grid_search_models(incremental_year):
#    models_processor.exhaustive_grid_search_models(interval = 1
#                                        , min_no_of_steps = 10
#                                        , max_no_of_steps = 20
#                                        , no_of_steps_interval = 10
#                                        , min_window_size = 5
#                                        , max_window_size = 10
#                                        , window_size_interval = 5
#                                        , interpolation_method = 'linear'
#                                        , train_start = (datetime.datetime.strptime('2007-01-01', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
#                                        , train_end = (datetime.datetime.strptime('2007-12-31', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
#                                        , test_start = (datetime.datetime.strptime('2008-01-01', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
#                                        , test_end = (datetime.datetime.strptime('2008-12-31', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
#                                        , dimensions = 3)


#Parallel(n_jobs=num_cores)(delayed(run_yearly_grid_search_models)(incremental_year) for incremental_year in incremental_years)



for incremental_year in range(3, 4):
    print(incremental_year)
        
    models_processor.exhaustive_grid_search_models(interval = 15
                                        , min_no_of_steps = 10
                                        , max_no_of_steps = 500
                                        , no_of_steps_interval = 10
                                        , min_window_size = 5
                                        , max_window_size = 80
                                        , window_size_interval = 5
                                        , interpolation_method = 'linear'
                                        , train_start = (datetime.datetime.strptime('2007-01-01', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
                                        , train_end = (datetime.datetime.strptime('2007-12-31', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
                                        , test_start = (datetime.datetime.strptime('2008-01-01', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
                                        , test_end = (datetime.datetime.strptime('2008-12-31', '%Y-%m-%d').date() + relativedelta(years=incremental_year)).strftime("%Y-%m-%d")
                                        , dimensions = 3)





#models_processor.exhaustive_grid_search_models(interval = 30
#                                    , min_no_of_steps = 5
#                                    , max_no_of_steps = 30
#                                    , no_of_steps_interval = 5
#                                    , min_window_size = 5
#                                    , max_window_size = 30
#                                    , window_size_interval = 5
#                                    , interpolation_method = 'linear'
#                                    , train_start = '2008-01-01'
#                                    , train_end = '2008-12-31'
#                                    , test_start = '2009-01-01'
#                                    , test_end = '2009-12-31'
#                                    , dimensions = 3)


