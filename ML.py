import pandas as pd
from sklearn.model_selection import train_test_split , KFold , GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("CollegePlacement.csv")

df = pd.DataFrame(data)

label_col = df[["Internship_Experience","Placement"]]

for col in label_col:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

x = df[["IQ","Prev_Sem_Result","CGPA","Academic_Performance","Internship_Experience",
        "Extra_Curricular_Score","Communication_Skills","Projects_Completed"]]
y = df["Placement"]


x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)

param_grid = {
    "criterion":['gini','entropy'],
    "max_depth":[2,3,4,5,6, None],
    "min_samples_split":[2,3,4,5,6]
}

KF = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=KF,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(x_train,y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)

print("Test Accuracy:",accuracy_score(y_test, y_pred))
