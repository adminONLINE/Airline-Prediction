import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
import joblib
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import joblib
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500000)
pd.set_option('display.max_rows', 30)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


######################################
# GÖREV 1 : Veri setine EDA işlemlerini uygulayınız.
######################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


################################################################
# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
################################################################

# train ve test setlerinin bir araya getirilmesi.

df_test = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")
df = pd.merge(df_test, df_train, how="outer")

# Unnamed: 0 ve id kolonunun silinmesi
df = df.drop(columns=["Unnamed: 0", "id"], axis=1)

# 'neutral or dissatisfied': 0, 'satisfied': 1 olarak veri setinin düzenlenmesi
df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)

######################################
# 1. Genel Resim
######################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
  #  print("##################### Quantiles #####################")
  #  print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

######################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

"""Bu analizde veri setinin dengesine aykırı bir durum olmadığı gözlemlenmiştir.  """

######################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

for col in num_cols:
    num_summary(df, col, True)

######################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
######################################

def target_summary_with_cat(dataframe, target, col_name):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(col_name)[target].value_counts()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"satisfaction",col)

def target_summary_with_num(dataframe, target, numeric_col):
    print(dataframe.groupby(target).agg({numeric_col: ["mean", "median"]}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"satisfaction",col)

# Bağımlı değişkenin incelenmesi
df["satisfaction"].hist(bins=100)
plt.show()

######################################
# 5. Korelasyon Analizi (Analysis of Correlation)
######################################

corr_num = df[num_cols].corr()
corr_num
# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr_num, cmap="RdBu")
plt.show()

""" Encode işleminden önce yapılan korelasyon anlam ifade etmedi."""


######################################
# Görev 2 : Feature Engineering
######################################

######################################
# Aykırı Değer Analizi
######################################

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    print(col, outlier_thresholds(df, col))

print(df["Age"].min(), df["Age"].max())
print(df["Flight Distance"].min(), df["Flight Distance"].max())
print(df["Departure Delay in Minutes"].min(), df["Departure Delay in Minutes"].max())
print(df["Arrival Delay in Minutes"].min(), df["Arrival Delay in Minutes"].max())

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

filtered_df = df[df['Arrival Delay in Minutes'] >= 455.0]
filtered_df2 = df['Arrival Delay in Minutes'] >= 200


""" Yapılan aykırı değer analizi sonrası aykırı değerlere müdahale edilmemesi gerektiğine karar verilmiştir 
çünkü "Departure Delay in Minutes" ve "Arrival Delay in Minutes" hedef değişkeni olması muhtemel şekilde etkilemiştir."""

######################################
# Eksik Değer Analizi
######################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df.isnull().sum()

""" "Arrival Delay in Minutes" değişkenindeki boş değerlerin ortalama ile doldurulması uygun görülmüştür. """

df.fillna(df["Arrival Delay in Minutes"].mean(), inplace=True)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "satisfaction", cat_cols)


######################################
# Yeni değişkenlerin oluşturulması.
######################################
# Yolcu yaşına göre segmentasyon
df["Total Delay in Minutes"] = abs(df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"])

# Yolcu yaşına göre segmentasyon
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 17, 30, 50, 70, 120], labels=['0-17', '18-30', '31-50', '51-70', '71+'])


#Uçuş mesafesine göre segmentasyon
df['Flight_Distance_Level'] = pd.cut(df['Flight Distance'], bins=[0, 1000, 3000, 5000, 10000, float('inf')],
                                       labels=['Short Haul', 'Medium Haul', 'Long Haul', 'Very Long Haul', 'Ultra Long Haul'])

#Uçuş gecikmesi durumları
df['Departure_Delay_Status'] = pd.cut(df['Departure Delay in Minutes'], bins=[-1, 0, 15, 60, 180, float('inf')],
                                         labels=['No Delay', 'Minor Delay', 'Moderate Delay', 'Significant Delay', 'Severe Delay'])
df['Arrival_Delay_Status'] = pd.cut(df['Arrival Delay in Minutes'], bins=[-1, 0, 15, 60, 180, float('inf')],
                                       labels=['No Delay', 'Minor Delay', 'Moderate Delay', 'Significant Delay', 'Severe Delay'])

#Toplam hizmet memnuniyeti skoru
service_columns = ['Inflight wifi service', 'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort',
                   'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                   'Inflight service', 'Cleanliness']
df['Total_Service_Satisfaction_Score'] = df[service_columns].sum(axis=1) / (len(service_columns))

# Inflight Service Satisfaction (Uçuş İçi Hizmet Memnuniyeti)
inflight_service_columns = ['Inflight entertainment','Leg room service','Baggage handling','Inflight service','Cleanliness']
df['Inflight_Service_Satisfaction'] = df[inflight_service_columns].sum(axis=1) / (len(inflight_service_columns))

# Online Services Satisfaction (Çevrimiçi Hizmet Memnuniyeti)
online_columns = ["Ease of Online booking", "Online boarding"]
df['Online_Services_Satisfaction_Score'] = df[online_columns].sum(axis=1) / (len(online_columns))

# Airport Service Satisfaction (Havaalanı İçi Hizmet Memnuniyeti)
airport_service_columns = ['On-board service', "Departure/Arrival time convenient", "Gate location", "Baggage handling", "Checkin service"]
df['Airport_Services_Satisfaction_Score'] = df[airport_service_columns].sum(axis=1) / (len(airport_service_columns))

# Inflight_Service_Satisfaction_Per_Mile (Uçuş İçi Mil Başına Hizmet Memnuniyeti)
df['Inflight_Service_Satisfaction_Per_Mile'] = df['Inflight_Service_Satisfaction'] / df["Flight Distance"]



cat_cols, cat_but_car, num_cols = grab_col_names(df)


##################
# Label Encoding & One-Hot Encoding işlemlerinin uygulanması.
##################

cat_cols, cat_but_car, num_cols = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols.remove("satisfaction")

new_cols = ['Gender','Customer Type','Type of Travel','Class','Age_Group','Flight_Distance_Level',
 'Departure_Delay_Status',
 'Arrival_Delay_Status']
df = one_hot_encoder(df, new_cols, drop_first=True)
df.columns

##################
# Korelasyon incelemesinin yapılması.
##################

corr = df.corr()

column_corr = corr['satisfaction'].drop('satisfaction')
plt.figure(figsize=(70, 20))  # Set figure size
plt.bar(column_corr.index, column_corr.values, color='skyblue')  # Create a bar plot

plt.title('Correlation with Column satisfaction')  # Set title
plt.xlabel('Variables')  # Set x-axis label
plt.ylabel('Correlation Coefficient')  # Set y-axis label
plt.xticks(rotation=270)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--')  # Add horizontal grid lines for readability

plt.show()


##################################
# MODELLEME
##################################

##################################
# GÖREV 3: Model kurma
##################################

y = df["satisfaction"]
X = df.drop(["satisfaction"], axis=1)

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

lgbm_model = lgb.LGBMClassifier(random_state=17)

lgbm_model.get_params()

# {'boosting_type': 'gbdt',
# 'class_weight': None,
# 'colsample_bytree': 1.0,
# 'importance_type': 'split',
# 'learning_rate': 0.1,
# 'max_depth': -1,
# 'min_child_samples': 20,
# 'min_child_weight': 0.001,
# 'min_split_gain': 0.0,
# 'n_estimators': 100,
# 'n_jobs': None,
# 'num_leaves': 31,
# 'objective': None,
# 'random_state': 17,
# 'reg_alpha': 0.0,
# 'reg_lambda': 0.0,
# 'subsample': 1.0,
# 'subsample_for_bin': 200000,
# 'subsample_freq': 0}

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.9625962426855559

cv_results['test_f1'].mean()
# 0.9562111539062533

cv_results['test_roc_auc'].mean()
# 0.9946743958193822

##################
# hiperparametre optimizasyonlarının gerçekleştirilmesi
##################

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(final_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.9645056975669849

cv_results['test_f1'].mean()
# 0.9584556199498984

cv_results['test_roc_auc'].mean()
# 0.9952014709992527



joblib.dump(final_model, 'kaan2.pkl')