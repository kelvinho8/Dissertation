
$cls

import sys
sys.path.append(r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")


from DB import DB
from Dataset import Dataset
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

df.head(16)


dataset = Dataset(data = db.query_to_dataframe(sql_string = sql), no_of_intervals_per_day = 8)

dataset.get_data()


dataset.get_data().head()

#dataset.get_data()[['Close', 'Volume']].iloc[:]


#dataset.derive_indicators()

#dataset.get_data().tail()

#dataset.visualize(columns=['Close', 'MA50', 'MA200'])



dataset.derive_features(no_of_steps = 1)



#dataset.get_data()

#upperband, middleband, lowerband = dataset.BB(10)

#close = dataset.get_data()['Close'].values

#type(close)


#dataset.talib2df(dataset.BB(30)).plot()
#dataset.get_data()['Close'].plot(secondary_y=True)

#plt.show()

#tran_bb = dataset.tran_BB(10)
#tran_bb[-1]


#for c, u, l, bb in zip(close[-100:-1], upperband[-100:-1], lowerband[-100:-1], tran_bb[-100:-1]):
#    print(c, u, l, bb)




#for c, u, l, bb in zip(close[1:100], upperband[1:100], lowerband[1:100], tran_bb[1:100]):
#    print(c, u, l, bb)


#type(tran_bb)

#dataset.talib2df(dataset.STOCH(30, 5)).plot()
#dataset.get_data()['STOCHKD30'].plot(secondary_y=True)

#plt.show()


dataset.data_cleaning()

dataset.get_data().tail()

dataset.get_data().head()['ReturnDummy'].unique()

dataset.get_data().head()

dataset.data_splitting(train_start = '2008-01-01', train_end = '2008-12-31', test_start = '2009-01-01', test_end = '2009-12-31')

dataset.get_trainset()

dataset.get_testset()

