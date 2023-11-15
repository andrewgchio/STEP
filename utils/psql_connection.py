# psql_connection.py

import psycopg2

# A context manager for a PostgreSQL database connection. 
# Use the context manager syntax: 
# 
#     with PostgreSQLConnection() as conn:
#         ...
# 
class PostgreSQLConnection:
    
    def __init__(self):
        '''
        Make a PostgreSQL connection and cursor
        '''
        self.user = 'postgres' # input('Enter PostgreSQL user[postgres]: ') or 'postgres'
        self.passwd = 'password' # getpass('Enter PostgreSQL password: ')
        
        
    def _login(self):
        '''
        Return a PostgreSQL connection using the credential attributes
        '''
        config = {
            'database' : 'sensorplacement',
            'user'     : self.user,
            'password' : self.passwd,
            'host'     : 'localhost',
            'port'     : '5432'
        }

        return psycopg2.connect(**config)
    
    def __enter__(self):
        '''
        Setup management for the context manager
        '''
        self.lcnx = self._login()
        self.cur  = self.lcnx.cursor()
        return self

    def cursor(self):
        return self.cur
    
    def commit(self):
        self.lcnx.commit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''
        Teardown management for the context manager
        '''
        self.lcnx.commit()
        self.lcnx.close()
        self.cur.close()
        return False
