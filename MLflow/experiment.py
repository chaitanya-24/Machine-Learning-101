import mlflow

def calculator(a,b,operation=None):
    if operation == 'addition':
        return a+b
    elif operation == 'subtraction':
        return a-b
    elif operation == 'multiplication':
        return a*b
    elif operation == 'division':
        return a//b
    


if __name__ == '__main__':
    with mlflow.start_run():
        a,b,operation = 10,20,'addition'
        mlflow.log_param('a', a)
        mlflow.log_param('b', b)
        mlflow.log_param('operation', operation)
        result = calculator(a,b,operation)
        print(result)
        mlflow.log_metric('result',result)
