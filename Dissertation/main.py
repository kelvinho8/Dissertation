
$cls

import sys
sys.path.append(r"C:\Users\Kelvin\CloudStation\MSC COMPUTER SCIENCE\Dissertation\CODE\Dissertation\Dissertation")


from DB import DB
from Dataset import Dataset


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

df.head()


dataset = Dataset(data = db.query_to_dataframe(sql_string = sql))

dataset.get_data()


dataset.get_data().head()

#dataset.get_data()[['Close', 'Volume']].iloc[:]


dataset.derive_indicators()

dataset.get_data().tail()

dataset.visualize(columns=['Close', 'MA50', 'MA250'])
