import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# regressor delcaration
regressor = linear_model.LinearRegression()


# data loading from sklearn
iris = datasets.load_iris()
#print(iris.DESCR)

# data representation
data = iris.data
target = iris.target
df = pd.DataFrame(data, columns = ["sepal length","sepal width","petal length","petal width"])
df = df.reset_index(drop = True)
dft = pd.DataFrame(target, columns = ["class"])
df_csv = pd.concat([df, dft], axis = 1)

#test train splitting
X_train, X_test, Y_train, Y_test = train_test_split(df, dft, test_size=0.2)

#performance metrics
regressor.fit(X_train, Y_train)
y = regressor.predict(X_test)
accuracy = regressor.score(X_test, Y_test)
print(accuracy)

#predicting new outputs
while(1 == True):
    sl = float(input("enter sepal length in cm"))
    sw = float(input("enter sepal width in cm"))
    pl = float(input("enter petal length in cm"))
    pw = float(input("enter petal width in cm"))

    in_array = np.array([sl, sw, pl, pw])
    predicted = regressor.predict([in_array])

    if int(predicted) == 0:
        print("Iris-Setosa")
    elif int(predicted) == 1:
        print("Iris-Versicolour")
    elif int(predicted)  == 2:
        print("Iris-Virginica")
    else:
        print("not recognized")
    print("--------------------------------------------------------------------------------------------------------") 