#İmporting Libraries(Kullanılacak Kütüphanelerin İmport Edilmesi) :

import numpy as np
import pandas as pd 
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import math as mt
import missingno as msno
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# İmport Data(Veri Setini İçe Aktarma):

def load_dataset():
  data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Examples/Titanik calisma/titanic.csv")
  return data

df = load_dataset()

titanic_df = df.copy()

titanic_df.head()

#Exploratory Data Analysis(Keşifçi Veri Analizi) :

#Firstly, we convert the column names to be upper.
#İlk olarak kolon isimlerini büyütüyoruz.
def upper_col_name(dataframe):
    upper_cols = [col.upper() for col in dataframe.columns]
    dataframe.columns = upper_cols
    return dataframe.head()
  
upper_col_name(titanic_df)

#The Function for Descriptive Statics(Veri Setine Ait Betimsel İstatistikler):

def check_df(dataframe, head = 5):
  print("//////////DataFrame Shape\\\\\\\\\\")
  print(dataframe.shape)
  print("//////////DataFrame Types\\\\\\\\\\")
  print(dataframe.dtypes)
  print("//////////DataFrame Head(First 5 rows) \\\\\\\\\\")
  print(dataframe.head(head))
  print("//////////DataFrame Tail(Last 5 rows) \\\\\\\\\\")
  print(dataframe.tail(head))
  print("//////////DataFrame NaN Values \\\\\\\\\\")
  print(dataframe.isnull().sum())
  print("//////////DataFrame Quantiles \\\\\\\\\\")
  print(dataframe.quantile([0,0.25,0.50,0.75,1]).T)
  
check_df(titanic_df)


#Grabbing to Categorical & Numarical Columns( Kategorik Ve Nümerik Veri Bloklarının Tutulması) :

def column_grab(dataframe, cat_th = 5, car_th = 20):
  
#Veri seti içindeki nümerik, kategorik ve nümerik görünümlü ordinal değişkenleri veri setinden seçmek adına tanımlanan bir fonksiyondur. Örneğin, Veri setindeki kadın ve erkek
#kategorik değişkenleri 0-1 olarak kategorize edilmiştir. Burada da görüleceği üzere nümerik değerler kullanılarak kategorik değerler oluşturulmuştur.

#cat_th : Kategorik özelliklere sahip sayısal değişkenlerdeki farklı gözlem sayısı için eşik değeri. 
#cat_th : Kategorik özelliklere sahip sayısal değişkenlerdeki farklı gözlem sayısı için eşik değeri. cat_th, sayısal değişkendeki farklı gözlemlerin sayısı 
#         cat_th'den azsa, bu değişkenlerin kategorik bir değişken olarak kategorize edilebileceğini belirtmek için kullanılır.




#car_th : Geniş bir kardinalite aralığına sahip kategorik değişkenler için eşik değeri.
#         Kategorik değişkenlerdeki farklı gözlemlerin sayısı car_th'den büyükse, bu değişken kategorik değişken olarak kategorize edilebilir.
 
#NOT : categorical_columns(cat_cols) + numerical_columns(num_cols) + cardinal_columns(cat_but_car) = Toplam Değişken Sayısı

#It is a function defined to select numeric, categorical and numeric-looking ordinal variables from the data set. For example, men and women in the dataset
#categorical variables are categorized as 0-1. As can be seen here, categorical values ​​were created using numerical values.

#cat_th : Threshold for the number of distinct observations in numeric variables with categorical properties.
#cat_th : Threshold for the number of distinct observations in numeric variables with categorical properties. cat_th, number of distinct observations in numeric variable
# If less than cat_th, it is used to indicate that these variables can be categorized as a categorical variable.

#car_th : Threshold for categorical variables with a wide range of cardinality.
# If the number of different observations in categorical variables is greater than car_th, this variable can be categorized as a categorical variable.
 
#NOTE : categorical_columns(cat_cols) + numerical_columns(num_cols) + cardinal_columns(cat_but_car) = Total Number of Variables
  
  #Kategorik ve kardinal değişkenler:
  cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
  
  num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and  dataframe[col].nunique() < cat_th]
                   
  cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
                   
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  # Nümerik değerler:
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and "ID" not in col.upper()]
  num_cols = [col for col in num_cols if col not in num_but_cat]
  return cat_cols,num_cols,cat_but_car


column_grab(titanic_df)

cat_cols, num_cols, cat_but_car = column_grab(titanic_df)

#General Exploration for Categorical Variables(Kategorik Değişken Özetleri):

def cat_summary(dataframe, plot = False):
    for col_name in cat_cols:
        print("############## Unique Observations of Categorical Data ###############")
        print("The unique number of "+ col_name+": "+ str(dataframe[col_name].nunique()))

        print("############## Frequency of Categorical Data ########################")
        print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                            "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))
        if plot == True:
            rgb_values = sns.color_palette("Set2", 6)
            sns.set_theme(style="darkgrid")
            ax = sns.countplot(x = dataframe[col_name], data = dataframe, palette= rgb_values)
            for p in ax.patches:
                ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=10)
            plt.show()
            
cat_summary(titanic_df, plot=True)

#General Exploration for Numerical Variables(Nümerik Değişken Özetleri):

def num_summary(dataframe, plot=False):
  quantiles = [0.25, 0.50, 0.75, 1]
  for col_name in num_cols:
    print("//////////// Summary Statistics of " + col_name + "\\\\\\\\\\\\" )
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
      sns.histplot(data = dataframe , x = col_name)
      plt.xlabel(col_name)
      plt.title("The Distribution of " + col_name)
      plt.grid(True)
      plt.show(block = True)
      
num_summary(titanic_df, plot=True)


#Feature Engineering( Değişken Etkileşimleri Yeni Değişken Oluşturma) :

#For passengers, only cabin information is available in the data set. Using the knowledge of the container 
#we create the Variable "New cabin" as a new variable for the passengers.
titanic_df["NEW_CABIN_CAT"] = titanic_df["CABIN"].notnull().astype('int')

#Using the name variable, we create the variable name_word_count, which gives the number of words that make up the name.
titanic_df["NEW_NAME_WORD_COUNT"] = titanic_df["NAME"].apply(lambda x: len(str(x).split(" ")))

## We're removing titles like DR before nouns.
titanic_df["NEW_NAME_DR"] = titanic_df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr. ")]))

#Number of Family Members:
titanic_df["NEW_FAMILY_SIZE"] = titanic_df["SIBSP"] + titanic_df["PARCH"] + 1

#Fare_per_person variable is the ticket price per passenger variable.
titanic_df['NEW_FARE_PER_PERSON'] = titanic_df['FARE'] / (titanic_df['NEW_FAMILY_SIZE'])

# Yine Name Sütunundan faydalanark isim basındaki diğer title bilgilerini ayrı bir sütun olark yazıyoruz.
titanic_df['NEW_TITLE'] = titanic_df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False) 

#For the Ticket variable, we generate a new ticket variable to get only the numeric parts.
titanic_df['NEW_TICKET'] = titanic_df['TICKET'].str.isalnum().astype('int')


#AGE_PCCLAS : This new variable column is a variable used for the interaction of younger but upper class passengers such as 1st class.
titanic_df["NEW_AGE_PCLASS"] = titanic_df["AGE"] * titanic_df["PCLASS"]


#Let's generate a new variable about whether the #passenger is alone or with his/her family.
titanic_df["NEW_IS_ALONE"] = np.where(titanic_df['SIBSP'] + titanic_df['PARCH'] > 0, "NO", "YES") 


#Let's generate a categorical variable for passengers using the age variable.
titanic_df.loc[(titanic_df['AGE'] < 18), 'NEW_AGE_CAT'] = 'Young'
titanic_df.loc[(titanic_df['AGE'] >= 18) & (titanic_df['AGE'] < 56), 'NEW_AGE_CAT'] = 'Mature'
titanic_df.loc[(titanic_df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'Senior'


#We generate a new categorical variable by crossing the age and gender variable.
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Male'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Female'


#Now that we have nothing to do with the PASSENGERID , NAME , TICKET and CABIN columns, we can drop them from the data set.
titanic_df.drop(columns=["PASSENGERID","NAME","TICKET","CABIN"], axis=1, inplace=True)




#Find outliers and lower-upper quartiles.
def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
    Q1 = dataframe[col_name].quantile(q1)
    Q3 = dataframe[col_name].quantile(q3)
    IQR = Q3 - Q1
    low_limit = Q1 - 1.5 * IQR
    up_limit = Q3 + 1.5 * IQR
    
    return low_limit, up_limit
  
cat_cols, num_cols, cat_but_car = column_grab(titanic_df)


for col in num_cols:
    print(col,":",outlier_thresholds(titanic_df,col))

 
#By visualizing the #missing values matrix, we can sketch whether there is a zero correlation or not. 
#As can be seen, there seems to be zero correlation for the variables NEW_AGE_PCLASS , NEW_AGE_CAT, NEW_SEX_CAT, AGE. 
#Let's examine the correlation with the heatmap carefully.
msno.matrix(titanic_df, figsize=(8,8), fontsize=8, labels=8)
plt.show()


msno.heatmap(titanic_df, figsize=(8,8), fontsize=12)
plt.show()


#Examination of observations with missing data:
def missing_values_df(dataframe, na_col_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    na_cols_number = dataframe[na_cols].isnull().sum()
    na_cols_ratio = dataframe[na_cols].isnull().sum() / dataframe.shape[0]
    missing_values_table = pd.DataFrame({"Missing_Values (#)": na_cols_number, \
                                         "Ratio (%)": na_cols_ratio * 100,
                                         "Type" : dataframe[na_cols].dtypes})
    print(missing_values_table)
    print("************* Number of Missing Values *************")
    print(dataframe.isnull().sum().sum())
    if na_col_name:
        print("************* Nullable variables *************")
        return na_cols
      
def missing_cat_cols_fill(dataframe):
    na_cols = [col for col in titanic_df.columns if titanic_df[col].isnull().sum() > 0 and titanic_df[col].dtype == "O"]
    for col in na_cols:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
        return dataframe.head()
      
#Now your categorical variables Sex, Embarked , Cabin etc. We group and aggregate the values of the Survived variable, which is our target variable, according to the variables.
def observe_missing_values(dataframe, na_col, related_col, target, target_method="mean", na_col_method="median"):
    print(dataframe.groupby(related_col).agg({target : target_method, 
                                               na_col : na_col_method}))
    
#We have aggregated the Survived values by grouping according to the Age variable.
#We see the statistics of the passengers who are alive or lost in the accident, according to their Embarked by Gender.

cat_cols = [col for col in cat_cols if col not in "SURVIVED"]
for col in cat_cols:
    observe_missing_values(titanic_df, "AGE",col,"SURVIVED")
    
#We can drop it from the data set as we are no longer working with the #NEW_NAME_DR variable.
titanic_df.drop(columns = "NEW_NAME_DR" , axis=1, inplace = True)


#We group the #Age(AGE) variable according to the title (groupby) and fill the missing values of the variable with the median of the Age variable. 
#In this way, we get rid of the problems related to the age variable.
titanic_df["AGE"] = titanic_df["AGE"].fillna(titanic_df.groupby("NEW_TITLE")["AGE"].transform("median"))


#We got rid of the #age variable, but the problem persists in the variables we derive using the age variable. So let's create these variables again.
#Age & Pclass
titanic_df["NEW_AGE_PCLASS"] = titanic_df["AGE"] * titanic_df["PCLASS"]

# Is Alone?
titanic_df["NEW_IS_ALONE"] = np.where(titanic_df['SIBSP'] + titanic_df['PARCH'] > 0, "NO", "YES")
    
# Age Level 
titanic_df.loc[(titanic_df['AGE'] < 18), 'NEW_AGE_CAT'] = 'Young'
titanic_df.loc[(titanic_df['AGE'] >= 18) & (titanic_df['AGE'] < 56), 'NEW_AGE_CAT'] = 'Mature'
titanic_df.loc[(titanic_df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'Senior'

 # Age & Sex
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Male'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Female'



#Outlier Detection with Local Outlier Factor(Local Outlier Factor İle Aykırı Gözlem Saptaması):
cat_cols, num_cols, cat_but_car = column_grab(titanic_df)
df_backup = titanic_df[num_cols]

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df_backup)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]


#LOF Visualization:
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

# Let's set the threshold value with the Elbow Method..
th = np.sort(df_scores)[8]

df_backup[df_scores < th]


#We drop outliers from the data set whose intensity value is less than the intensity value of the determined threshold values.
titanic_df.drop(df_backup[df_scores < th].index, inplace=True)



#Standardizing Data - Encoding (Verileri Standartlaştırma - Kodlama):

def binary_cols(dataframe):
  binary_col_names = [col for col in dataframe.columns if ((dataframe[col].dtype == "O") and (dataframe[col].nunique() == 2))]
  return binary_col_names
binary_col_names = binary_cols(titanic_df)


#Label Encoding :
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
  
for col in binary_col_names:
    label_encoder(titanic_df, col)
    

def rare_analyser(dataframe, target):
    cat_cols, num_cols, cat_but_car = column_grab(dataframe)
    cat_cols = [col for col in cat_cols if  col != target in cat_cols]

    for col in cat_cols:
        print(col, ":", dataframe[col].nunique())
        print("dtype:", dataframe[col].dtype)
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(), \
                            "RATIO (%)": dataframe[col].value_counts() / dataframe.shape[0], \
                            "TARGET_MEAN (%) ": dataframe.groupby(col)[target].mean() * 100}))
      
rare_analyser(titanic_df, "SURVIVED")


# Rare Encoder :

def rare_encoder(dataframe, rare_perc=0.0100):
    rare_df = dataframe.copy()

    rare_columns = [col for col in rare_df.columns if rare_df[col].dtypes == 'O'
                    and (rare_df[col].value_counts() / rare_df.shape[0] <= rare_perc).any(axis=None)]

    for col in rare_columns:
        tmp = rare_df[col].value_counts() / rare_df.shape[0]
        rare_labels = tmp[tmp <= rare_perc].index
        rare_df[col] = np.where(rare_df[col].isin(rare_labels), 'Rare', rare_df[col])

    return rare_df
  
new_titanic_df = rare_encoder(titanic_df)

#Here we check to see if rare column still exist.
rare_analyser(new_titanic_df, "SURVIVED")


#We made determinations about rare variables. Now let's drop the observations less than 1% from our data set.
def useless_cols(dataframe, rare_perc=0.01):
    useless_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2
                    and (dataframe[col].value_counts() / len(dataframe) <= rare_perc).any(axis=None)]
    new_df = dataframe.drop(useless_cols, axis=1)
    return useless_cols 

useless_cols(new_titanic_df)
#We saw that any variable-column was not seen as useless.
  
  
#One - Hot Encoding :

#One-Hot encoder allows us to perform binarization of these categorical data.

def ohe_cols(dataframe):
    ohe_cols = [col for col in dataframe.columns if (dataframe[col].dtype == "O" and 10 >= dataframe[col].nunique() > 2)]
    return ohe_cols
  
ohe_col_names = ohe_cols(new_titanic_df)

def one_hot_encoder(dataframe, ohe_col_names, drop_first=True):
    dms = pd.get_dummies(dataframe[ohe_col_names], drop_first=drop_first)    
    df_ = dataframe.drop(columns=ohe_col_names, axis=1)              
    dataframe = pd.concat([df_, dms],axis=1)                     
    return dataframe

new_titanic_df = one_hot_encoder(new_titanic_df, ohe_col_names)


#Standardization :
#Standardization is a method in which the mean value is 0 and the standard deviation is 1, and the distribution approaches normal. 
#The formula is as follows, we subtract the mean value from the value we have, then divide by the variance value.
cat_cols, num_cols, cat_but_car = column_grab(new_titanic_df)

scaler = StandardScaler()
new_titanic_df[num_cols] = scaler.fit_transform(new_titanic_df[num_cols])



#Correlation Charts:

#In this part, we visualize to see the level of correlation between the variables.
#In addition, we include the ones in the upper triangle matrix, whose correlation value is greater than the specified correlation threshold value (0.75), in a separate list as a drop list.

def high_correlated_cols(dataframe, plot=False, corr_th=0.75):
    cat_cols, num_cols, cat_but_car = column_grab(dataframe)
    cor_matrix = dataframe[num_cols].corr().abs()
    #corr = dataframe.corr()
    #cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (8, 8)})
        sns.set(font_scale=1) 
        sns.heatmap(cor_matrix, cmap="RdBu",annot=True)
        plt.show()
    return drop_list
  
high_correlated_cols(new_titanic_df, plot=True)
  
  
drop_list = ["FARE","SIBSP","PARCH"]
  
new_titanic_df = new_titanic_df.drop(drop_list, axis=1)
  
#In this section, since different variables are derived depending on the "AGE" variable (NEW_SEX_CAT , NEW_AGE_CAT etc.)
#the modeling will be started by excluding the "AGE" variable.
new_titanic_df = new_titanic_df.drop(columns="AGE",axis=1)
  
  
  
#Modelling(Modelleme) :
  
#Logistic Regression Model(Lojistik Regresyon Modeli) :
  
X = new_titanic_df.drop(columns="SURVIVED",axis=1)
y = new_titanic_df[["SURVIVED"]]

# We split the data set as test and train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=112)

# Training the model:
log_model = LogisticRegression().fit(X_train,y_train)

# Prediction ( Tahminleme )
y_pred = log_model.predict(X_test)


confusion_matrix(y_test, y_pred)

#Test Matrix of Model Performance(Model Performansının Test Matrisi):
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Precision Score :", precision_score(y_test,y_pred))
print("Recall Score:" ,recall_score(y_test,y_pred))
print("F1 Score:", f1_score(y_test,y_pred))


#AUC stands for “Area under the ROC Curve”.
#The scope of this field is AUC. The larger the area covered, the better the machine learning models are at distinguishing given classes. The ideal value for AUC is 1.
AUC = logit_roc_auc =roc_auc_score(y_test,y_pred)
plt.figure(figsize=(6,6))
fpr ,tpr,thresholds= roc_curve(y_test,log_model.predict_proba(X_test)[:,1])
plt.plot(fpr,tpr,label ="AUC (area=%0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.savefig("Log_ROC")
plt.show();

#Cross - Validation(Çapraz Doğrulama):
#Cross-validation is a model validation technique that tests the results of a statistical analysis on an independent data set.
cross_val_score(log_model, X_test,y_test,cv=10,scoring= "neg_mean_squared_error")
np.mean(cross_val_score(log_model, X_test,y_test,cv=10))


#Feature Importance:
feature_importance = pd.DataFrame(X_train.columns, columns = ["feature"])
feature_importance["importance"] = pow(mt.e, log_model.coef_[0])
feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
 
# Visualization 
ax = feature_importance.plot.barh(x='feature', y='importance', figsize=(8,8), fontsize=10)
plt.xlabel('Önem Düzeyi', fontsize=14)
plt.ylabel('Değişkenler', fontsize=14)
plt.show()

feature_importance[0:10]

new_features = feature_importance[0:10]
cols = [col for col in new_features["feature"]]



X_ = new_titanic_df[cols]
y_ = new_titanic_df[["SURVIVED"]]


X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.20, random_state=112)
                                                     
log_model_ = LogisticRegression().fit(X_train_,y_train_)

y_pred_ = log_model_.predict(X_test_)

confusion_matrix(y_test_, y_pred_)


print("Accuracy Score:",accuracy_score(y_test_,y_pred_))
print("Precision Score:", precision_score(y_test_,y_pred_))
print("Recall Score:" ,recall_score(y_test_,y_pred_))
print("F1 Score:", f1_score(y_test_,y_pred_))

#ROC CURVE 2

AUC = logit_roc_auc =roc_auc_score(y_test_,y_pred_)

fpr ,tpr,thresholds= roc_curve(y_test_,log_model_.predict_proba(X_test_)[:,1])
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label ="AUC (area=%0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.savefig("Log_ROC")
plt.show();



# Model Validation 

cross_val_score(log_model_, X_test_,y_test_,cv=10,scoring= "neg_mean_squared_error")
#print(cross_val_score(log_model, X_test,y_test,cv=10))    
np.mean(cross_val_score(log_model_, X_test_,y_test_,cv=10))


feature_importance = pd.DataFrame(X_train_.columns, columns = ["feature"])
feature_importance["importance"] = pow(mt.e, log_model_.coef_[0])
feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
 
# Visualization 
ax = feature_importance.plot.barh(x='feature', y='importance', figsize=(8,8), fontsize=10)
plt.xlabel('Önem Düzeyi', fontsize=14)
plt.ylabel('Değişkenler', fontsize=14)
plt.show()








