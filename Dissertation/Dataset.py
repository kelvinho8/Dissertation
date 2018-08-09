class Dataset(object):
    def __init__(self, data):
        self.__data = data

    def get_data(self):
        return self.__data

    def derive_indicators(self):
        from talib import abstract

        input_arrays = {
            'open': self.__data['Open'].values,
            'high': self.__data['High'].values,
            'low': self.__data['Low'].values,
            'close': self.__data['Close'].values,
            'volume': self.__data['Volume'].values
        }

        self.__data['MA10'] = abstract.SMA(input_arrays, timeperiod=10)
        self.__data['MA20'] = abstract.SMA(input_arrays, timeperiod=20)
        self.__data['MA30'] = abstract.SMA(input_arrays, timeperiod=30)
        self.__data['MA50'] = abstract.SMA(input_arrays, timeperiod=50)
        self.__data['MA100'] = abstract.SMA(input_arrays, timeperiod=100)
        self.__data['MA200'] = abstract.SMA(input_arrays, timeperiod=200)
        self.__data['MA250'] = abstract.SMA(input_arrays, timeperiod=250)
        
    def visualize(self, columns):
        import matplotlib.pyplot as plt

        self.__data[columns].plot()
        plt.show()









