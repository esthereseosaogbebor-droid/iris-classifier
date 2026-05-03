import os

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data   # shape (150, 4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
labels = iris.target_names.tolist() # ['setosa', 'versicolor', 'virginica']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
out_dir ="output"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "confusion_matrix.png")
fig .savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Confusion matrix saved to: {out_path}")
model_path = os.path.join(out_dir, "decision_tree_iris.pkl")
joblib.dump(model, model_path, compress=3)
print(f"trained model saved to: {model_path}")