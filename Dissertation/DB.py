class DB(object):
    def __init__(self, driver, server, database, username, password):
        self.driver = driver
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection_string = 'DRIVER='+self.driver+';SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password


    def delete_table(self, sql_string):
        import pyodbc
        import pandas as pd

        con = pyodbc.connect(self.connection_string)
        cursor = con.cursor()
        cursor.execute(sql_string)

        con.commit()
        con.close()



    def read_sql(self, sql_string):
        import pyodbc
        import pandas as pd

        con = pyodbc.connect(self.connection_string)
        df = pd.read_sql(sql_string, con)
        
        df.index = df['TradingDatetime']
        del df['TradingDatetime']
        con.close()

        return df

    def to_sql(self, df, table_name, if_exists):
        #import pyodbc
        #import pandas as pd
        #import sqlalchemy
        import urllib
        from sqlalchemy import create_engine

        params = urllib.parse.quote_plus(self.connection_string)

        db = create_engine('mssql+pyodbc:///?odbc_connect=%s' % params)

        df.to_sql(name = table_name, con = db, if_exists = if_exists, index=False)
        db.dispose()





