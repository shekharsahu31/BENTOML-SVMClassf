import bentoml

from sklearn import svm
from sklearn import datasets


iris = datasets.load_iris()

X,y = iris.data , iris.target

clf = svm.SVC(gamma = 'scale')
clf.fit(X,y)

#save model to bentoml local model store
saved_model = bentoml.sklearn.save_model("iris_clf" , clf)
print(f"Model saved: {saved_model}")

# iris_clf:pxdvcsmm5w4nraer
