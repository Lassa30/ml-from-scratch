# yeah... it will be longer in c++ :-)

# type: ignore
import pandas as pd 

iris_df = pd.read_csv('Iris.csv')
iris_df = iris_df.iloc[:, 1:]

classes_dict = {'iris-setosa': 0,
                'iris-versicolor': 1,
                'iris-virginica': 2 }

iris_df.iloc[:, -1] = (iris_df.iloc[:, -1]).apply(lambda string : classes_dict[string.lower()])

iris_df.to_csv('IrisModified.csv', index=False, header=False)
