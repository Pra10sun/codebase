import os
from sqlalchemy.engine.url import URL

LOG_LEVEL = 'INFO'

db_connection_credentials = {
    'drivername': 'postgresql+pg8000',
    'username': 'researcher',
    'password': 'cb6a042a-7693-4e12-9135-8abb1dbf3f7f',
    'host': os.environ['RDS_DEVELOP_ENDPOINT'],
    'port': 5432,
    'database': 'postgres'
}

