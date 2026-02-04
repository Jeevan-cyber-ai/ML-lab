"""import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
data = {
    'Hours': [2, 4, 5, 6, 8],
    'Attendance': [60, 65, 70, 75, 85],
    'Result': [0, 0, 1, 1, 1]   
}

df = pd.DataFrame(data)
X = df[['Hours', 'Attendance']]
y = df['Result']
model = RandomForestClassifier(
    n_estimators=10,     
    criterion='gini',     
    random_state=42
)


model.fit(X, y)
test_data = pd.DataFrame([[5, 72]], columns=['Hours', 'Attendance'])
prediction = model.predict(test_data)

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")

from sklearn import tree

plt.figure(figsize=(12,6))
tree.plot_tree(
    model.estimators_[0],
    feature_names=['Hours', 'Attendance'],
    class_names=['Fail', 'Pass'],
    filled=True
)
plt.show()"""



import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(iris.target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
new_sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)

prediction = model.predict(new_sample)

print("Predicted class:", iris.target_names[prediction[0]])

sklearn.tree.plot_tree(model.estimators_[0])
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importance
importances = model.feature_importances_

# Create DataFrame
feature_imp = pd.Series(
    importances,
    index=iris.feature_names
).sort_values(ascending=False)

# Plot
plt.figure(figsize=(8,5))
feature_imp.plot(kind='bar')
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.title("Feature Importance - Random Forest (Iris)")
plt.show()
