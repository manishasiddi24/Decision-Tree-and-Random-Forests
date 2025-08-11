import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import graphviz

# Load dataset (example: Heart Disease from UCI repo)
data = pd.read_csv("heart.csv")
X, y = data.drop("target", axis=1), data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Visualize Tree
dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns, 
                           class_names=["No Disease", "Disease"], filled=True)
graphviz.Source(dot_data).render("decision_tree", format="png", cleanup=True)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", importances)

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Random Forest CV Accuracy:", cv_scores.mean())