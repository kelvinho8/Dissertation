class Dataset(object):
    def __init__(self, data, no_of_intervals_per_day, no_of_steps):
        self.data = data
        self.no_of_intervals_per_day = no_of_intervals_per_day

        if no_of_steps < 1:
            self.no_of_steps = 1
        else:
            self.no_of_steps = no_of_steps




    def get_data(self):
        return self.data



    def get_no_of_steps(self):
        return self.no_of_steps



    def get_return(self):
        return self.data['Close'].pct_change(self.no_of_steps)

    def get_return_dummy(self):
        import numpy as np
        #return np.sign(self.data['Close'].pct_change(self.no_of_steps))

        return self.data['Close'].pct_change(self.no_of_steps).to_frame().applymap(lambda x: 1 if x > 0 else 0).iloc[:]


    def MA(self, window_size):
        from talib import abstract

        input_arrays = {
            'open': self.data['Open'].values,
            'high': self.data['High'].values,
            'low': self.data['Low'].values,
            'close': self.data['Close'].values,
            'volume': self.data['Volume'].values
        }

        return abstract.SMA(input_arrays, timeperiod=window_size * self.no_of_intervals_per_day)
        #return abstract.SMA(input_arrays, timeperiod=window_size)

    def tran_MA(self, window_size):
        close = self.data['Close'].values
        ma = self.MA(window_size)

        return (close - ma) / ma



    def BB(self, window_size):
        from talib import abstract

        input_arrays = {
            'open': self.data['Open'].values,
            'high': self.data['High'].values,
            'low': self.data['Low'].values,
            'close': self.data['Close'].values,
            'volume': self.data['Volume'].values
        }

        return abstract.BBANDS(input_arrays, timeperiod=window_size * self.no_of_intervals_per_day, nbdevup=2, nbdevdn=2, matype=0)
        #return abstract.BBANDS(input_arrays, timeperiod=window_size, nbdevup=2, nbdevdn=2, matype=0)
        

    def tran_BB(self, window_size):
        import numpy as np

        close = self.data['Close'].values
        upperband, middleband, lowerband = self.BB(window_size)

        bb_feature = []
        for c, u, l in zip(close, upperband, lowerband):
            if c <= u and c >= l:
                bb_feature.append(0)
            elif c > u:
                bb_feature.append(c - u)
            elif c < l:
                bb_feature.append(c - l)
            else:
                bb_feature.append(float('nan'))


        return np.asarray(bb_feature)

    
    def RSI(self, window_size):
        from talib import abstract

        input_arrays = {
            'open': self.data['Open'].values,
            'high': self.data['High'].values,
            'low': self.data['Low'].values,
            'close': self.data['Close'].values,
            'volume': self.data['Volume'].values
        }

        return abstract.RSI(input_arrays, timeperiod=window_size * self.no_of_intervals_per_day)
        #return abstract.RSI(input_arrays, timeperiod=window_size)


    def tran_RSI(self, window_size):
        close = self.data['Close'].values
        rsi = self.RSI(window_size)

        return (rsi - 50) / 50


    def STOCH(self, window_size):
        from talib import abstract

        input_arrays = {
            'open': self.data['Open'].values,
            'high': self.data['High'].values,
            'low': self.data['Low'].values,
            'close': self.data['Close'].values,
            'volume': self.data['Volume'].values
        }
            
        #slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        #return abstract.STOCH(input_arrays, fastk_period=window_size * self.no_of_intervals_per_day, slowk_period=window_size * self.no_of_intervals_per_day, slowk_matype=0, slowd_period=window_size * self.no_of_intervals_per_day, slowd_matype=0)
    
        return abstract.STOCH(input_arrays, fastk_period=window_size * self.no_of_intervals_per_day, slowk_period=3 * self.no_of_intervals_per_day, slowk_matype=0, slowd_period=3 * self.no_of_intervals_per_day, slowd_matype=0)
        #return abstract.STOCH(input_arrays, fastk_period=window_size, slowk_period=window_size, slowk_matype=0, slowd_period=window_size, slowd_matype=0)
    

    #def STOCH(self, window_size_fastk, window_size_slowk, window_size_slowd):
    #    from talib import abstract

    #    input_arrays = {
    #        'open': self.data['Open'].values,
    #        'high': self.data['High'].values,
    #        'low': self.data['Low'].values,
    #        'close': self.data['Close'].values,
    #        'volume': self.data['Volume'].values
    #    }
            
    #    #slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    #    return abstract.STOCH(input_arrays, fastk_period=window_size_fastk * self.no_of_intervals_per_day, slowk_period=window_size_slowk * self.no_of_intervals_per_day, slowk_matype=0, slowd_period=window_size_slowd * self.no_of_intervals_per_day, slowd_matype=0)
    


    def tran_STOCH_K(self, window_size):
        
        slowk, slowd = self.STOCH(window_size)

        return (slowk - 50) / 50

    def tran_STOCH_KD(self, window_size):
        
        slowk, slowd = self.STOCH(window_size)

        return ((slowk - slowd) - 50) / 50


    def remove_na(self):
        self.data.dropna(inplace=True)

    def interpolate(self, method = 'linear'):
        self.data.interpolate(method, inplace = True)


    def data_splitting(self, train_start, train_end, valid_start, valid_end, test_start, test_end):
        self.train = self.data[train_start:train_end]
        self.valid = self.data[valid_start:valid_end]
        self.test = self.data[test_start:test_end]

    def get_train(self):
        return self.train

    def get_valid(self):
        return self.valid

    def get_test(self):
        return self.test

    def avoid_look_ahead_bias(self):
        self.train = self.train[1:-self.no_of_steps]


    def set_X(self, features):
        self.X = features

    def set_y(self, feature):
        self.y = feature

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def set_X_train(self):
        self.X_train = self.train[self.X]

    def set_y_train(self):
        self.y_train = self.train[self.y]

    def set_X_valid(self):
        self.X_valid = self.valid[self.X]

    def set_y_valid(self):
        self.y_valid = self.valid[self.y]

    def set_X_test(self):
        self.X_test = self.test[self.X]

    def set_y_test(self):
        self.y_test = self.test[self.y]



    def get_X_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train
    
    def get_X_valid(self):
        return self.X_valid

    def get_y_valid(self):
        return self.y_valid
    
    def get_X_test(self):
        return self.X_test

    def get_y_test(self):
        return self.y_test


    
    def get_y_train_true(self):
        return self.y_train.sum()

    def get_y_train_false(self):
        return self.y_train.count() - self.y_train.sum()

    def get_y_train_ratio(self):
        return self.y_train.sum() * 1.0 / self.y_train.count()


    def get_y_valid_true(self):
        return self.y_valid.sum()

    def get_y_valid_false(self):
        return self.y_valid.count() - self.y_valid.sum()

    def get_y_valid_ratio(self):
        return self.y_valid.sum() * 1.0 / self.y_valid.count()


    def get_y_test_true(self):
        return self.y_test.sum()

    def get_y_test_false(self):
        return self.y_test.count() - self.y_test.sum()

    def get_y_test_ratio(self):
        return self.y_test.sum() * 1.0 / self.y_test.count()



    def get_X_train_start_date(self):
        return self.X_train.index[0]
    
    def get_X_train_end_date(self):
        return self.X_train.index[-1]

    def get_X_valid_start_date(self):
        return self.X_valid.index[0]
    
    def get_X_valid_end_date(self):
        return self.X_valid.index[-1]


    def get_X_test_start_date(self):
        return self.X_test.index[0]

    def get_X_test_end_date(self):
        return self.X_test.index[-1]



    def print_train_test_period(self):        
        print(self.X_train.index[0], self.X_train.index[-1], self.X_valid.index[0], self.X_valid.index[-1], self.X_test.index[0], self.X_test.index[-1])
        print('ValidRatio = {}, Y_Valid.True = {}, T_Valid.False = {}\n'.format(self.get_y_valid_ratio(), self.get_y_valid_true(), self.get_y_valid_false()))
        print('TestRatio = {}, Y_Test.True = {}, T_Test.False = {}\n'.format(self.get_y_test_ratio(), self.get_y_test_true(), self.get_y_test_false()))

    #def print_train_test_period(self):        
    #    return print(self.X_train.index[0], self.X_train.index[-1], self.X_test.index[0], self.X_test.index[-1])



    def normalization(self):
        from sklearn import preprocessing
        import pandas as pd

        scaler = preprocessing.StandardScaler().fit(self.X_train)

        self.X_train = pd.DataFrame(data = scaler.transform(self.X_train), columns = self.X, index = self.train.index)
        self.X_valid = pd.DataFrame(data = scaler.transform(self.X_valid), columns = self.X, index = self.valid.index)
        self.X_test =  pd.DataFrame(data = scaler.transform(self.X_test), columns = self.X, index = self.test.index)


    def dimension_reduction(self, n_components=2):
        from sklearn.decomposition import PCA
        import pandas as pd
        import numpy as np

        pca = PCA(n_components, svd_solver='full')
        pca.fit(self.X_train)
        print(pca.explained_variance_)
        print(pca.explained_variance_ratio_)
        print(np.sum(pca.explained_variance_ratio_))

        self.X_train = pd.DataFrame(data = pca.transform(self.X_train), index = self.train.index)
        self.X_valid =  pd.DataFrame(data = pca.transform(self.X_valid), index = self.valid.index)
        self.X_test =  pd.DataFrame(data = pca.transform(self.X_test), index = self.test.index)




    def derive_features(self, window_size = 10):
        self.data['Return'] = self.get_return()
        self.data['ReturnDummy'] = self.get_return_dummy()

        
        self.data['MA' + str(window_size)] = self.tran_MA(window_size)
        self.data['MA' + str(window_size * 2)] = self.tran_MA(window_size * 2)
        self.data['MA' + str(window_size * 3)] = self.tran_MA(window_size * 3)
        #self.data['MA' + str(window_size * 4)] = self.tran_MA(window_size * 4)
        #self.data['MA' + str(window_size * 5)] = self.tran_MA(window_size * 5)

        self.data['BB' + str(window_size)] = self.tran_BB(window_size)
        self.data['BB' + str(window_size * 2)] = self.tran_BB(window_size * 2)
        self.data['BB' + str(window_size * 3)] = self.tran_BB(window_size * 3)

        self.data['RSI' + str(window_size)] = self.tran_BB(window_size)
        self.data['RSI' + str(window_size * 2)] = self.tran_BB(window_size * 2)
        self.data['RSI' + str(window_size * 3)] = self.tran_BB(window_size * 3)

        self.data['STOCHK' + str(window_size)] = self.tran_STOCH_K(window_size = window_size)
        self.data['STOCHK' + str(window_size * 2)] = self.tran_STOCH_K(window_size = window_size * 2)
        self.data['STOCHK' + str(window_size * 3)] = self.tran_STOCH_K(window_size = window_size * 3)

        self.data['STOCHKD' + str(window_size)] = self.tran_STOCH_KD(window_size = window_size)
        self.data['STOCHKD' + str(window_size * 2)] = self.tran_STOCH_KD(window_size = window_size * 2)
        self.data['STOCHKD' + str(window_size * 3)] = self.tran_STOCH_KD(window_size = window_size * 3)

        #print(self.data.head())
        

    def visualize(self, columns):
        import matplotlib.pyplot as plt

        self.data[columns].plot()
        plt.show()

    def talib2df(self, talib_output):
        import pandas as pd

        if type(talib_output) == list:
            ret = pd.DataFrame(talib_output).transpose()
        else:
            ret = pd.Series(talib_output)
        ret.index = self.data['Close'].index
        return ret;








