class Backtester(object):
    """description of class"""
    def __init__(self, interval = 30):
        import sys
        sys.path.append(r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")
        #sys.path.append(r"C:\Users\Kelvi\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")

        import pandas as pd
        from collections import OrderedDict
        import pytz

        from DB import DB
        
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

        data = OrderedDict()
        ticker = 'SH00300'
        df = db.read_sql(sql_string = sql)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.names = ['datetime']
        print(df.head())

        data[ticker] = df
        data[ticker] = data[ticker][['open', 'high', 'low', 'close', 'volume']]
        print(data[ticker].head())

        #data[ticker]['datetime'] = pd.to_datetime(data[ticker]['datetime'], unit='s')
        print(data[ticker].index)
        print(type(data[ticker].index))

        data[ticker].index = data[ticker].index.tz_localize(pytz.timezone("Asia/Shanghai")).tz_convert(pytz.timezone("UTC"))
        
        print(data[ticker].index)
        print(type(data[ticker].index))


        self.panel = pd.Panel(data)
        self.panel.minor_axis = ['open', 'high', 'low', 'close', 'volume']
        #self.panel.marjo_axis = self.panel.major_axis.tz_localize(pytz.timezone("Asia/Shanghai"))
        print(self.panel)

        #data[ticker] = pd.read_csv('{}.CSV'.format(ticker), index_col = 0, parse_dates = ['date'])
        #data[ticker] = data[ticker][['open', 'high', 'low', 'close', 'volume']]
        #print(data[ticker].head())
    
        #panel = pd.Panel(data)
        #panel.minor_axis = ['open', 'high', 'low', 'close', 'volume']
        #panel.marjo_axis = panel.major_axis.tz_localize(pytz.utc)
        #print(panel)

    def run_algorithm(self):
        from zipline.api import order, record, symbol, set_benchmark
        import zipline
        import matplotlib.pyplot as plt
        from datetime import datetime
        import pytz
        from zipline.utils.calendars.exchange_calendar_shsz import SHSZExchangeCalendar

        def initialize(context):
            set_benchmark(symbol('SH00300'))


        def handle_data(context, data):
            order(symbol('SH00300'), 10)
            record(SPY=data.current(symbol('SH00300'), 'price'))

        self.perf = zipline.run_algorithm(start=datetime(2017, 3, 6, 9, 31, 0, 0, pytz.timezone("Asia/Shanghai")),
                                      end=datetime(2017, 3, 9, 3, 30, 0, 0, pytz.timezone("Asia/Shanghai")),
                                      initialize=initialize,
                                      trading_calendar=SHSZExchangeCalendar(),
                                      capital_base=100000,
                                      handle_data=handle_data,
                                      data_frequency='minute',
                                      data=self.panel)
        self.perf.head()


    def visualize(self):
        import matplotlib.pyplot as plt

        self.perf.portfolio_value.plot()
        plt.show()

