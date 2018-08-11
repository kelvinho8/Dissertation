class Dataset(object):
    def __init__(self, data, no_of_intervals_per_day):
        self.__data = data
        self.__no_of_intervals_per_day = no_of_intervals_per_day


    def get_data(self):
        return self.__data

    def get_return(self, no_of_steps = 1):
        if no_of_steps < 1:
            return self.__data['Close'].pct_change(1)
        else:
            return self.__data['Close'].pct_change(no_of_steps)

    def get_return_dummy(self, no_of_steps = 1):
        import numpy as np
        if no_of_steps < 1:
            return np.sign(self.__data['Close'].pct_change(1))
        else:
            return np.sign(self.__data['Close'].pct_change(no_of_steps))



    def MA(self, window_size):
        from talib import abstract

        input_arrays = {
            'open': self.__data['Open'].values,
            'high': self.__data['High'].values,
            'low': self.__data['Low'].values,
            'close': self.__data['Close'].values,
            'volume': self.__data['Volume'].values
        }

        return abstract.SMA(input_arrays, timeperiod=window_size * self.__no_of_intervals_per_day)


    def tran_MA(self, window_size):
        close = self.__data['Close'].values
        ma = self.MA(window_size)

        return (close - ma) / ma



    def BB(self, window_size):
        from talib import abstract

        input_arrays = {
            'open': self.__data['Open'].values,
            'high': self.__data['High'].values,
            'low': self.__data['Low'].values,
            'close': self.__data['Close'].values,
            'volume': self.__data['Volume'].values
        }

        return abstract.BBANDS(input_arrays, timeperiod=window_size, nbdevup=2, nbdevdn=2, matype=0)


    def tran_BB(self, window_size):
        import numpy as np

        close = self.__data['Close'].values
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
            'open': self.__data['Open'].values,
            'high': self.__data['High'].values,
            'low': self.__data['Low'].values,
            'close': self.__data['Close'].values,
            'volume': self.__data['Volume'].values
        }

        return abstract.RSI(input_arrays, timeperiod=window_size * self.__no_of_intervals_per_day)



    def tran_RSI(self, window_size):
        close = self.__data['Close'].values
        rsi = self.RSI(window_size)

        return (rsi - 50) / 50


    def STOCH(self, window_size_k, window_size_d):
        from talib import abstract

        input_arrays = {
            'open': self.__data['Open'].values,
            'high': self.__data['High'].values,
            'low': self.__data['Low'].values,
            'close': self.__data['Close'].values,
            'volume': self.__data['Volume'].values
        }
            
        #slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

        return abstract.STOCH(input_arrays, fastk_period=window_size_k * self.__no_of_intervals_per_day, slowk_period=window_size_d * self.__no_of_intervals_per_day, slowk_matype=0, slowd_period=window_size_d * self.__no_of_intervals_per_day, slowd_matype=0)
    


    def tran_STOCH_K(self, window_size_k, window_size_d):
        
        slowk, slowd = self.STOCH(window_size_k, window_size_d)

        return (slowk - 50) / 50

    def tran_STOCH_KD(self, window_size_k, window_size_d):
        
        slowk, slowd = self.STOCH(window_size_k, window_size_d)

        return ((slowk - slowd) - 50) / 50


    def data_cleaning(self):
        self.__data.dropna(inplace=True)



    def derive_features(self, no_of_steps = 1):
        self.__data['Return'] = self.get_return(no_of_steps)
        self.__data['ReturnDummy'] = self.get_return_dummy(no_of_steps)

        
        self.__data['MA10'] = self.tran_MA(10)
        self.__data['MA20'] = self.tran_MA(20)
        self.__data['MA30'] = self.tran_MA(30)
        self.__data['MA40'] = self.tran_MA(40)
        self.__data['MA50'] = self.tran_MA(50)

        self.__data['BB10'] = self.tran_BB(10)
        self.__data['BB20'] = self.tran_BB(20)
        self.__data['BB30'] = self.tran_BB(30)

        self.__data['RSI10'] = self.tran_BB(10)
        self.__data['RSI20'] = self.tran_BB(20)
        self.__data['RSI30'] = self.tran_BB(30)

        self.__data['STOCHK10'] = self.tran_STOCH_K(window_size_k = 10, window_size_d = 5)
        self.__data['STOCHK20'] = self.tran_STOCH_K(window_size_k = 20, window_size_d = 5)
        self.__data['STOCHK30'] = self.tran_STOCH_K(window_size_k = 30, window_size_d = 5)

        self.__data['STOCHKD10'] = self.tran_STOCH_KD(window_size_k = 10, window_size_d = 5)
        self.__data['STOCHKD20'] = self.tran_STOCH_KD(window_size_k = 20, window_size_d = 5)
        self.__data['STOCHKD30'] = self.tran_STOCH_KD(window_size_k = 30, window_size_d = 5)

        

    def visualize(self, columns):
        import matplotlib.pyplot as plt

        self.__data[columns].plot()
        plt.show()

    def talib2df(self, talib_output):
        import pandas as pd

        if type(talib_output) == list:
            ret = pd.DataFrame(talib_output).transpose()
        else:
            ret = pd.Series(talib_output)
        ret.index = self.__data['Close'].index
        return ret;








