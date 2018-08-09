class DB(object):
    def __init__(self, driver, server, database, username, password):
        self.driver = driver
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection_string = 'DRIVER='+self.driver+';SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password

    def query_to_dataframe(self, sql_string):
        import pyodbc
        import pandas as pd

        con = pyodbc.connect(self.connection_string)
        df = pd.read_sql(sql_string, con)
        
        df.index = df['TradingDatetime']
        del df['TradingDatetime']
        con.close()

        return df




