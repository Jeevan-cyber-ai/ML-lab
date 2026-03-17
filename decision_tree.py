import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = {
    'Hours': [2, 4, 5, 6, 8],
    'Attendance': [60, 65, 70, 75, 85],
    'Result': [0, 0, 1, 1, 1]   
}

df = pd.DataFrame(data)
model = DecisionTreeClassifier(criterion='gini')
X = df[['Hours', 'Attendance']]
y = df['Result']

model.fit(X, y)
test_data = pd.DataFrame([[5, 72]], columns=['Hours', 'Attendance'])
prediction = model.predict(test_data)

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")

plt.figure(figsize=(10,6))
tree.plot_tree(
    model,
    feature_names=['Hours', 'Attendance'],
    class_names=['Fail', 'Pass'],
    filled=True
)
plt.show()
