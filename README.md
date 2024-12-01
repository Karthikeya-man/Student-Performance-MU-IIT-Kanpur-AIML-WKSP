import pandas as pd
dataset_link = "https://raw.githubusercontent.com/Ranjit-Singh-786/MU/refs/heads/master/Student_Performance.csv"
df = pd.read_csv(dataset_link)
df.info()
df.shape
# to check the duplicates recods
no_of_duplicates_rrecord = df.duplicated().sum()
print("Number of duplicate records:", no_of_duplicates_rrecord)
# to delete the duplicate records
df.drop_duplicates(inplace=True)
df.shape
df.isnull().sum()          # to check the missing records
df
df['Extracurricular Activities'].unique()
df['Extracurricular Activities'].value_counts()
Extracurricular_Activities_dt = {'Yes': 1, 'No': 0}
df['Extracurricular Activities'] = df['Extracurricular Activities'].map(Extracurricular_Activities_dt)
df
import matplotlib.pyplot as plt
import seaborn as sns
Hours_Studied = df['Hours Studied'].value_counts().keys()
Hours_Studied
Previous_Scores = df['Previous Scores'].value_counts().keys()
Previous_Scores
Hours_Studied = [2, 3, 5, 6, 8, 10, 12, 14, 16]  # Length 9
Previous_Scores = [50, 55, 60, 65, 70, 75, 80, 85, 90]  # Length 9

# Check lengths
if len(Hours_Studied) != len(Previous_Scores):
    # Truncate the longer list to match the shorter one
    min_length = min(len(Hours_Studied), len(Previous_Scores))
    Hours_Studied = Hours_Studied[:min_length]
    Previous_Scores = Previous_Scores[:min_length]

# Plot bar chart
plt.bar(Hours_Studied, Previous_Scores, color='green')
plt.xticks(rotation=90)
plt.title("Analysis")
plt.ylabel("Previous Scores")
plt.xlabel("Hours Studied")
plt.show()

plt.pie(Hours_Studied, labels=Previous_Scores, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title("Analysis")
plt.show()
plt.scatter(Hours_Studied, Previous_Scores)
plt.title("Analysis")
plt.xlabel("Hours Studied")
plt.ylabel("Previous Scores")
plt.show()
plt.hist(Hours_Studied, bins=10, color='green', edgecolor='black')
plt.title("Analysis")
plt.xlabel("Hours Studied")
plt.ylabel("Frequency")
plt.show()
sns.kdeplot(Hours_Studied, color='green', shade=True)
plt.title("Analysis")
plt.xlabel("Hours Studied")
plt.ylabel("Frequency/Density")
plt.show()

sns.lineplot(x=Hours_Studied, y=Previous_Scores, color='green')
plt.title("Analysis")
plt.xlabel("Hours Studied")
plt.ylabel("Previous Scores")
plt.show()
df['Hours Studied'].unique()
dt2 = {'2': 2, '3': 3, '5': 5, '6': 6, '8': 8, '10': 10, '12': 12, '14': 14, '16': 16}
print(dt2)
df['Hours Studied'] = df['Hours Studied'].map(dt2)
df.drop(["Hours Studied","Previous Scores"],axis="columns",inplace=True)
df.head()
x = df[["Extracurricular Activities"]]
y = df[["Performance Index"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
lnr = LinearRegression()
dtr  =  DecisionTreeRegressor()
rdf = RandomForestRegressor()
x_train.shape
lnr.fit(x_train,y_train)
dtr.fit(x_train,y_train)
rdf.fit(x_train,y_train)
print("successfully trained the algorithms!")
print(lnr.score(x_train,y_train))
print(dtr.score(x_train,y_train))
print(rdf.score(x_train,y_train))
print(lnr.score(x_test,y_test))
print(dtr.score(x_test,y_test))
print(rdf.score(x_test,y_test))
x_test.shape
prediction = rdf.predict(x_test)
y_test['prediction'] = prediction
y_test.head(40)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(x_train, y_train)
y_pred_adaboost = adaboost.predict(x_test)
!pip install xgboost
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from xgboost import XGBClassifier
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train.values.ravel())
xgboost = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgboost.fit(x_train, y_train_encoded)
y_pred_xgboost = xgboost.predict(x_test)
y_pred_original = le.inverse_transform(y_pred_xgboost)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)


