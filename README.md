## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="354" height="450" alt="image" src="https://github.com/user-attachments/assets/1099eb75-542e-43cf-b06c-9e8a84531c07" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="208" height="256" alt="image" src="https://github.com/user-attachments/assets/38c6c9da-6a56-4a5d-8d7b-abe07a967368" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="408" height="448" alt="image" src="https://github.com/user-attachments/assets/dbb4bbdc-a111-4672-b068-4c76f49fd5ae" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="391" height="453" alt="image" src="https://github.com/user-attachments/assets/1d9dbdc0-6029-4071-8eff-f28341c36873" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="512" height="456" alt="image" src="https://github.com/user-attachments/assets/6d11e272-68d7-4f4b-b678-0538256f60fd" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="509" height="443" alt="image" src="https://github.com/user-attachments/assets/5e2aa968-aad1-429e-aba9-b2c049365ffb" />

pip install --upgrade category_encoders

<img width="1904" height="462" alt="image" src="https://github.com/user-attachments/assets/b9c3f439-06fd-4611-8cdd-25ee20f81f38" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

<img width="585" height="456" alt="image" src="https://github.com/user-attachments/assets/e654be1c-be54-4bd3-831a-9b3f62d8ca0f" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="837" height="450" alt="image" src="https://github.com/user-attachments/assets/c30e8a2e-2db2-45d9-9d0e-b5c51bb7c4a6" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="848" height="460" alt="image" src="https://github.com/user-attachments/assets/2cda4240-4fe8-4dca-9c03-a7e980f61847" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

<img width="961" height="507" alt="image" src="https://github.com/user-attachments/assets/2a868210-538d-4ae1-a33a-50b91bc6ddbc" />

df.skew()

<img width="354" height="265" alt="image" src="https://github.com/user-attachments/assets/cb64484a-9c62-4fbb-9b12-a9266b18b60d" />

np.log(df["Highly Positive Skew"])

<img width="311" height="524" alt="image" src="https://github.com/user-attachments/assets/67431520-22b9-4183-aca4-0b325fdc18e2" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="337" height="531" alt="image" src="https://github.com/user-attachments/assets/ee7abd1c-bef7-48b3-81d3-1de6fe9991e0" />

np.sqrt(df["Highly Positive Skew"])

<img width="325" height="522" alt="image" src="https://github.com/user-attachments/assets/d8444fb6-6a8e-425d-b468-1d20727db59e" />

np.square(df["Highly Positive Skew"])

<img width="310" height="510" alt="image" src="https://github.com/user-attachments/assets/1122eca9-f583-4376-9cd5-c41ff4119382" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1236" height="525" alt="image" src="https://github.com/user-attachments/assets/7f5585b8-13f9-44ea-adb6-ef37bce3954b" />

df.skew()

<img width="399" height="248" alt="image" src="https://github.com/user-attachments/assets/4eb6160e-b2c7-4a8d-aaf9-cb222c9c61c3" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="466" height="312" alt="image" src="https://github.com/user-attachments/assets/fabc9408-fb9f-4985-a264-ea40a95f6139" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1798" height="510" alt="image" src="https://github.com/user-attachments/assets/a5d5e823-65e5-403e-b816-2421b6225994" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```

<img width="755" height="573" alt="image" src="https://github.com/user-attachments/assets/cab07160-c8ac-4923-aac7-d8f247b966a9" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```

<img width="735" height="561" alt="image" src="https://github.com/user-attachments/assets/a7dd2459-0290-463a-bf9a-88ccf3e904ff" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="720" height="568" alt="image" src="https://github.com/user-attachments/assets/6c179160-1c8c-446e-8bef-337e45811526" />


# RESULT:
Thus the cammands are executed successfully.
       
