import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier" , runners = [iris_clf_runner])
# runners -> list of models we can give in order like 
# transformers , pipelines and etc and in last models. 

@svc.api(input=NumpyNdarray() , output = NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result

# Using this function we can directly create our models/model into an API
# without any extra loads 
# np.ndarray - datatype - numpy array
# in background swagger.api is used