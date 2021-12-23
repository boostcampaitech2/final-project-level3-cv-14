'''
database를 초기화합니다.
'''
import toml
import mysql.connector

secrets = toml.load("Utils/streamlit/secrets.toml")

# serving_database 데이터베이스 생성 
conn =  mysql.connector.connect(
    host=secrets['mysql']['host'],
    user=secrets['mysql']['user'],
    passwd=secrets['mysql']['password'],
)

if conn is None:
    assert "conn is empty"

with conn.cursor() as cur:
    try:
        cur.execute("DROP DATABASE serving_database")    
    except:
        pass
    cur.execute("CREATE DATABASE serving_database")

conn.close()

# serving_database 데이터베이스안에 INPUT, INFERENCE, SCORE 테이블 생성
conn =  mysql.connector.connect(
    host=secrets['mysql']['host'],
    user=secrets['mysql']['user'],
    passwd=secrets['mysql']['password'],
    database=secrets['mysql']['database'],
)


with conn.cursor() as cur:
    try:
        sql = "DROP TABLE "
        cur.execute(sql+"INPUT")
        cur.execute(sql+"INFERENCE")
        cur.execute(sql+"MASK")
        cur.execute(sql+"SCORE")
    except:
        pass
    cur.execute("""
    CREATE TABLE INPUT 
    (
        input_id VARCHAR(36) PRIMARY KEY,
        input_url VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE INFERENCE 
    (
        id INT AUTO_INCREMENT PRIMARY KEY,
        input_id VARCHAR(36),
        inference_url VARCHAR(255),
        inference_type VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE SCORE 
    (
        id INT AUTO_INCREMENT PRIMARY KEY,
        input_id VARCHAR(36),
        score VARCHAR(10)
    )
    """)
    cur.execute("""
    CREATE TABLE MASK 
    (
        id INT AUTO_INCREMENT PRIMARY KEY,
        input_id VARCHAR(36),
        mask_url VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

conn.close()