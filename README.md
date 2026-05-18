# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
df = pd.read_csv("bmi.csv")  
print("Original Dataset:")
print(df.head())

df = df.dropna()

df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print("\nStandard Scaled Data:")
print(df_std.head())
```


<img width="411" height="331" alt="Screenshot 2026-05-17 184011" src="https://github.com/user-attachments/assets/ad1b3a0a-572d-4530-8879-b0045feb91af" />



```
df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())

```



<img width="426" height="201" alt="Screenshot 2026-05-17 184101" src="https://github.com/user-attachments/assets/c19f2ca3-e5e1-40ae-85e0-3b957b660d4c" />




```
df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())
```



<img width="412" height="181" alt="Screenshot 2026-05-17 184203" src="https://github.com/user-attachments/assets/0e7cda32-6bdb-4cf6-8d15-dff1f0a0594d" />




```
df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())
```



<img width="407" height="176" alt="Screenshot 2026-05-17 184725" src="https://github.com/user-attachments/assets/d9b532e7-5e40-4d0d-bd17-5728c43b769c" />



```
df_std.to_csv("BMI_StandardScaled.csv", index=False)
df_minmax.to_csv("BMI_MinMaxScaled.csv", index=False)
df_maxabs.to_csv("BMI_MaxAbsScaled.csv", index=False)
df_robust.to_csv("BMI_RobustScaled.csv", index=False)

print("\nFeature Scaling Completed Successfully.")
```


<img width="425" height="47" alt="Screenshot 2026-05-17 193133" src="https://github.com/user-attachments/assets/8c673079-8079-4b75-ba56-4d8c2d80d6ea" />




```
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df = pd.read_csv("income(1) (1).csv")
print("Dataset Preview:")
print(df.head())
```



<img width="835" height="481" alt="Screenshot 2026-05-17 193230" src="https://github.com/user-attachments/assets/c6b52511-962d-4173-ad99-7e41517fd458" />





```
#Encode Cateorigal Variables
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation','relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes
X = df.drop(columns=['SalStat'])
y = df['SalStat']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))
```




<img width="1012" height="58" alt="Screenshot 2026-05-17 193408" src="https://github.com/user-attachments/assets/97d1aa31-6c4a-4ed6-a70f-d5553415db04" />




```
selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))
```




<img width="800" height="53" alt="Screenshot 2026-05-17 193458" src="https://github.com/user-attachments/assets/456d28e0-d5fe-4c0e-95a4-9f9a092c8a5f" />



```
logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))
```




<img width="902" height="51" alt="Screenshot 2026-05-17 193551" src="https://github.com/user-attachments/assets/2fff9a1e-a714-4274-8179-fb355c0d0ce5" />



```
rf = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf.fit(X_train, y_train)

selector_embedded = SelectFromModel(rf, threshold="mean")
selector_embedded.fit(X_train, y_train)

selected_features_embedded = X.columns[selector_embedded.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))
```



<img width="1023" height="53" alt="Screenshot 2026-05-17 193710" src="https://github.com/user-attachments/assets/0055c05e-f646-4bc8-a221-aecbe06cf844" />



```
X_train_sel = selector_embedded.transform(X_train)
X_test_sel = selector_embedded.transform(X_test)

rf.fit(X_train_sel, y_train)
y_pred = rf.predict(X_test_sel)

print("\nModel Accuracy (Embedded Method):", accuracy_score(y_test, y_pred))
```




<img width="522" height="37" alt="Screenshot 2026-05-17 193804" src="https://github.com/user-attachments/assets/b1f95497-84b2-4674-aa23-65357179705d" />


# RESULT:
      Thus the Feature Scaling and selection Executed successfully.

