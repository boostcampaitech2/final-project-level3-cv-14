'''
- 이미지 정보를 INPUT, INFERENCE, SCORE 테이블에 입력
- INPUT, INFERENCE 테이블로부터 target_date에 해당하는 주차의 데이터를 불러옴
'''
import toml
import mysql.connector


def _execute(qry, val, is_select=False):
    """ 쿼리 실행
    Args:
    - qry: SQL QUERY
    - val: COLUMN NAME
    - is_select : True이면 SELECT, False이면 INSERT
    """
    secrets = toml.load("WebServer/secret/secrets.toml")
    conn =  mysql.connector.connect(
        host=secrets['mysql']['host'],
        user=secrets['mysql']['user'],
        passwd=secrets['mysql']['password'],
        database=secrets['mysql']['database']
        )
    
    with conn.cursor(dictionary=True) as cur: 
        cur.execute(qry, val)
        if is_select:
            result = cur.fetchall()
        else:
            conn.commit()
            result = cur.lastrowid 

    conn.close()
    return result


def insert_data_input(input_id, input_url):
    qry = "INSERT INTO INPUT (input_id, input_url) VALUES (%s, %s)"
    val = (input_id, input_url)
    _ = _execute(qry, val, is_select=False)
    

def insert_data_inference(input_id, inference_url, inference_type):
    qry = "INSERT INTO INFERENCE (input_id, inference_url, inference_type) VALUES (%s, %s, %s)"
    val = (input_id, inference_url, inference_type)
    _ = _execute(qry, val, is_select=False)
    
    
def insert_data_score(input_id, score):
    qry = "INSERT INTO SCORE (input_id, score) VALUES (%s, %s)"
    val = (input_id, score)
    _ = _execute(qry, val, is_select=False)
    

def get_week_data_input(target_date): 
    """target_date 주에 해당하는 주차 데이터를 불러오기. (from INPUT table)
    """
    qry = "SELECT * FROM INPUT WHERE EXTRACT(WEEK FROM created_at) = EXTRACT(WEEK FROM (%s))"
    val = (target_date, )
    return _execute(qry, val, is_select=True)


def get_week_data_inference(target_date): 
    """target_date 주에 해당하는 주차 데이터를 불러오기. (from INFERENCE table)
    """
    qry = "SELECT * FROM INFERENCE WHERE EXTRACT(WEEK FROM created_at) = EXTRACT(WEEK FROM %s)"
    val = (target_date, )
    return _execute(qry, val, is_select=True)

