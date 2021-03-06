from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator

from airflow.operators.python_operator import PythonOperator

def print_hello():
    return 'Hello world!'

#dag = DAG('hello_world', description='Simple tutorial DAG',
#          schedule_interval='0 12 * * *',
#          start_date=datetime(2019, 6, 04), catchup=False)
'''
default_args = {
		'owner' : 'Micah',
		'depends_on_past' :False,
		'email' :['micah@investanalyze.com'],
		'email_on_failure': False,
		'email_on_retry': False,
		'retries': 5,
		'retry_delay': timedelta(minutes=1)
		}
'''
#run_this_first = DummyOperator(
#    task_id='run_this_first',
#    dag=dag,
#)

dag = DAG('hello_world', description='Simple tutorial DAG',
          schedule_interval='0 12 * * *',
          start_date=datetime(2017, 3, 20), catchup=False)

dummy_operator = DummyOperator(task_id='dummy_task', retries=3, dag=dag)
#dag = DAG(	dag_id='hello_world')
# here dag_id = python file name
#dummy_operator = DummyOperator(task_id='dummy_task', dag=dag)
#dummy_operator = BashOperator(task_id='dummy_task', retries=3, dag=dag)

hello_operator = PythonOperator(task_id='hello_task', python_callable=print_hello, dag=dag)

dummy_operator >> hello_operator
