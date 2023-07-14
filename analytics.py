#========================== Dataset Description ===============================
"""
Description Of the DataSets:

This csv file contains information like Country, other names,
ISO 3166-1 alpha-3 CODE, Population, Continent, Total Cases, Total Deaths,
Tot Cases//1M pop, Tot Deaths/1M pop, Death percentage of COVID 19 Coronavirus pandemic

"""
import pandas as pd
data = pd.read_csv("COVID-19 Coronavirus.csv")
data.describe()
#======================= Data Visualization ===================================

#------(1) Continent Death Percentag
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("COVID-19 Coronavirus.csv")

data_africa=data.loc[data['Continent']=="Africa"]
#africa_=data_africa['Total_Cases'].sum()
africa_deaths=data_africa['Total_Deaths'].sum()
africa_cases=data_africa['Total_Cases'].sum()

data_Europe=data.loc[data['Continent']=="Europe"]
Europe_deaths=data_Europe['Total_Deaths'].sum()
Europe_cases=data_Europe['Total_Cases'].sum()

data_LatinAmerica=data.loc[data['Continent']=="Latin America and the Caribbean"]
LatinAmerica_deaths=data_LatinAmerica['Total_Deaths'].sum()
LatinAmerica_cases=data_LatinAmerica['Total_Cases'].sum()

data_Asia=data.loc[data['Continent']=="Asia"]
Asia_deaths=data_Asia['Total_Deaths'].sum()
Asia_cases=data_Asia['Total_Cases'].sum()

data_Oceania=data.loc[data['Continent']=="Oceania"]
Oceania_deaths=data_Oceania['Total_Deaths'].sum()
Oceania_cases=data_Oceania['Total_Cases'].sum()

data_NorthernAmerica=data.loc[data['Continent']=="Northern America"]
NorthernAmerica_deaths=data_NorthernAmerica['Total_Deaths'].sum()
NorthernAmerica_cases=data_NorthernAmerica['Total_Cases'].sum()

Deaths=[africa_deaths,Europe_deaths,LatinAmerica_deaths,Asia_deaths,Oceania_deaths,NorthernAmerica_deaths]
print(Deaths)
labels=["Africa","Europe","Latin America","Asia","Oceania","Northern America"]
myexplode = [0, 0.1, 0, 0,0.3,0]
plt.pie(Deaths, labels = labels,explode = myexplode,autopct='%1.1f%%',shadow=True)
plt.title('Continent Virus Deaths Percentage')
plt.axis('equal')
plt.show()

"""
Observation : Europe is the most contient affected with the virus
while oceania and africa are the least
"""

#------(2) HealthCare
rate=[round((africa_deaths/africa_cases),3),
(round((Europe_deaths/Europe_cases),3)),
(round((LatinAmerica_deaths/LatinAmerica_cases),3)),
(round((Asia_deaths/Asia_cases),3)),
(round((Oceania_deaths/Oceania_cases),3)),
(round((NorthernAmerica_deaths/NorthernAmerica_cases),3))]
print(rate)
print(africa_deaths/africa_cases)
labels=["Africa","Europe","Latin America","Asia","Oceania","Northern America"]
plt.plot(labels,rate,color='red', marker='o')
plt.title('Death Rate Vs Place', fontsize=14)
plt.xlabel('Healthcare in latin america and africa is low ', fontsize=14)
plt.ylabel('Death Rate', fontsize=14)
plt.grid(True)
plt.show()
"""
Observation : Africa and Latin America has the largest deaths from the total cases has the virus
and that mean that this Contient has bad health care while Europe and Northern America has
the best heath care and dealing with the Epidemic well
"""


#-------(3)top 20 countries covid deaths & least 20 countries covid deaths
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("COVID-19 Coronavirus.csv")

largest_countries=data.nlargest(20,'Total_Deaths')
largest_countries=largest_countries.sort_values(by=['Total_Deaths'])

fig, ax = plt.subplots(figsize=(23, 10))
ax.ticklabel_format(style='plain')
sns.barplot(largest_countries['Country'],largest_countries['Total_Deaths'])
plt.title('top 20 countries covid deaths',fontsize=25)

ndata=data.loc[data['Total_Deaths']> 0]
smallest_countries=ndata.nsmallest(20,'Total_Deaths')
fig, ax = plt.subplots(figsize=(35, 15))
ax.ticklabel_format(style='plain')
sns.barplot(smallest_countries['Country'],smallest_countries['Total_Deaths'])
plt.title('Smallest 20 countries covid deaths',fontsize=25)

plt.show()

"""
Observation: USA and brazil have the largest  number of deaths
while samoa and western sahara are the lowst  number of deaths
"""


#------(4) plot show Seriousness Virus Visualization
labels=["Africa","Europe","Latin America","Asia","Oceania","Northern America"]
x = np.arange(len(labels))  # the label locations

df = pd.DataFrame({'a': [africa_deaths,Europe_deaths,LatinAmerica_deaths,Asia_deaths,Oceania_deaths,NorthernAmerica_deaths],
                   'b': [africa_cases, Europe_cases, LatinAmerica_cases, Asia_cases,Oceania_cases,NorthernAmerica_cases]})
fig, ax1 = plt.subplots(figsize=(10, 7))
plt.ticklabel_format(style='plain') # to prevent scientific notation.

df['b'].plot(kind='bar', color='c')
df['a'].plot(kind='line', marker='d',color='r')
plt.xticks(x, ['Africa', 'Europe', 'Latin America', 'Asia', 'Oceania','Northern America'])
plt.xlabel("Continents")
plt.ylabel("cases")
plt.title('Virus Deaths is very small ', fontsize=18)
plt.legend(["Deaths", "cases"],prop = {'size' : 20})

"""
Observation: deaths of virus doesn't exceed 2% from the cases
so we can coexist with this virus
"""


#========================== Anomaly Detection ================================


#-----(1)
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from numpy import random, where
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("COVID-19 Coronavirus.csv")

f4=data['Total_Cases'].values
f5=data['Total_Deaths'].values

X=np.array(list(zip(f4,f5)))
print(X)
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
print(X)

x=pd.DataFrame(X,columns=["Total_Cases","Total_Deaths"])

print(x)
x.plot.scatter(x="Total_Cases", y="Total_Deaths", s=40)

dbscan = DBSCAN(eps = 0.28, min_samples =3,metric="euclidean")
pred =dbscan .fit_predict(x)
print(pred)
cmap=cm.get_cmap("Set1")
x.plot.scatter(x="Total_Cases", y="Total_Deaths", c=pred ,cmap=cmap,colorbar=False,s=40)


#-----(2)
data = pd.read_csv("COVID-19 Coronavirus.csv")
data['DeathPercentage']=(data['Total_Deaths']/data['Total_Cases'])
Q3=data['DeathPercentage'].describe()['75%']
Q1= data['DeathPercentage'].describe()['25%']
DeathPercentage_IQR= Q3 - Q1
above_Outliers=Q3+1.5*DeathPercentage_IQR
below_Outliers=Q1-1.5*DeathPercentage_IQR
col =[]
OutlierCountriees=[]
index=[]
DC=data['Country']
DB=data['DeathPercentage'].values
for i in range(0, 225):
    if DB[i]>=above_Outliers or DB[i]<=below_Outliers :
        col.append('blue')
        index.append(i)
        OutlierCountriees.append(DC[i])

    else:
        col.append('red')

print(index)
fig, ax = plt.subplots()
for i in range(225):
    if i not in index:
        a=ax.scatter(i,DB[i], c ='deepskyblue', s = 10)
    else:
        b=ax.scatter(i,DB[i], c ='red', s = 15,marker='x')
plt.xlabel("Countries")
plt.ylabel("Death percentage")
plt.title('Outliers values of Death Percentage ', fontsize=18)
ax.legend((a,b),('Normal value','Outliers value'))
ax.grid(True)
plt.show()
print("Population IQR : ",DeathPercentage_IQR)
print("Outliers Range below and above :","{",below_Outliers," , ",above_Outliers,"}")
print("Outliers countries in Death Percentage is :",'\n',OutlierCountriees)


#-----(3)
data = pd.read_csv("COVID-19 Coronavirus.csv")

deathpercentage=data['Total_Deaths']/data['Total_Cases']
ax = sns.boxplot(x=deathpercentage,y=data['Continent'])
# adding transparency to colors
for patch in ax.artists:
 r, g, b, a = patch.get_facecolor()
 patch.set_facecolor((r, g, b, .9))

plt.show()


#(4)
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from numpy import random, where
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv(r"COVID-19 Coronavirus.csv")

totalCases = np.array(data['Total_Deaths'])
print(totalCases)
random.seed(7)
x, _ = make_blobs(n_samples=200, centers=1, cluster_std=.3, center_box=(20, 40))
plt.xlabel("Total_Deaths")
plt.ylabel("Death percentage")
dbscan = DBSCAN(eps = 0.28, min_samples = 15)
pred =dbscan .fit_predict(x)
anom_index = where(pred == -1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='r')
plt.show()



#===================== predictive analytic techniques==========================


#------------(1)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as sm

data = pd.read_csv("COVID-19 Coronavirus.csv")
part_df = data[["Total_Deaths", "Total_Cases"]]

c=data['Country']
X=data[["Population",'Total_Cases']].values
y=data['Total_Deaths']
linr_model = LinearRegression(fit_intercept=(False)).fit(X, y)
prediction=linr_model.predict(X)
for i in range(len(prediction)):
    print(c[i],':   ','Total Cases',X[i,0],'  ',
          'Predicted Death',int(prediction[i]),'  ','Actual Death:',y[i])

print("########################")
print(  "Cofficent is : ", linr_model.coef_)
print("Intercept is : ", linr_model.intercept_)
print("\n","R2 Score =", round(sm.r2_score(y, prediction), 2))
pred=[]
for i in range(len(prediction)):
    pred.append(int(prediction[i]))

#2d plotting
fig, ax1 = plt.subplots(figsize=(10, 7))
plt.ticklabel_format(style='plain') # to prevent scientific notation.
plt.plot(y,c='red')
plt.plot(prediction,c='blue')
plt.legend(['Actual','Predicted'],fontsize=18)
plt.xlabel("Multiple Regression of Population and Total Cases")
plt.ylabel("Total Deaths")
plt.grid(True)
plt.show()

#3d plotting
ax = plt.axes(projection ='3d')
# defining all 3 axes
z = data["Population"]
x = data['Total_Cases']
y = data['Total_Deaths']
p=prediction
plt.ticklabel_format(style='plain') # to prevent scientific notation.
ax.plot3D(x, y, z, 'c')
ax.plot3D(x, p, z, 'r')
ax.set_xlabel('Total Cases', labelpad=20)
ax.set_ylabel('Total_Deaths', labelpad=20)
ax.set_zlabel('Population', labelpad=20)
plt.legend(['Actual','Predicted'],fontsize=10)
plt.grid(True)
ax.set_title('3D line plot ')
plt.show()


#------------(2)
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import sklearn.metrics as sm

data = pd.read_csv("COVID-19 Coronavirus.csv")
part_df = data[["Total_Deaths", "Total_Cases"]]
c=data['Country']
X=data[["Population",'Total_Cases']].values
y=data['Total_Deaths']
model=DecisionTreeRegressor()
model=model.fit(X, y)
prediction=model.predict(X)
for i in range(len(prediction)):
    print(c[i],':   ','Total Cases',X[i,0],'  ',
          'Predicted Death',int(prediction[i]),'  ','Actual Death:',y[i])

print("\n","R2 Score =", round(sm.r2_score(y, prediction), 2))
print("\n","Population :",model.feature_importances_[0]*100,"%","\n","Total_Cases :",
      model.feature_importances_[1]*100,"%","\n")



#--------------(3)
import sklearn.metrics as sm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv("Fifa_players.csv")
#part_df = data[["Total_Deaths", "Total_Cases"]]

c=data['Name']
X=data[["Age","Potential","Total stat"]].values
y=data["Overall"]
Model = RandomForestRegressor(n_estimators=100).fit(X, y)
prediction=Model.predict(X)
for i in range(len(prediction)):
    if  int(prediction[i]) > 85:
        print(c[i],':   ','Predicted overall',int(prediction[i]),'  ','Actual overall:',y[i])

print("\n","R2 Score =", round(sm.r2_score(y, prediction), 2))
print("\n","age :",Model.feature_importances_[0]*100,"%","\n","Potential :",
      Model.feature_importances_[1]*100,"%","\n",
      "Total stat :",Model.feature_importances_[2]*100,"%")

plt.plot(y,marker='x',color='r')
plt.plot(prediction,marker='o',color='c')
plt.title("Actual and Predicted Overall (fifa players)")
plt.xlabel("Number of Players ")
plt.ylabel("Players Rate")
plt.legend(['Actual','Predicted'])
plt.show()



#============================= Text Mining ====================================

#(1) StopWords ,Tokeniztion ,stremming and Lemmatization
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


#StopWords and Tokeniztion
stop_words = set(stopwords.words('english'))
with open('text.txt') as f:
    contents = f.readlines()

for sentence in contents:
    #remove punctuation
    sentence_without_Punc=sentence.translate(str.maketrans('','',string.punctuation))

    #tokenize
    tokens = word_tokenize(sentence_without_Punc)
    tokens_result = [i for i in tokens if not i in stop_words]
    print('\n','Sentence is:','\n',sentence,'\n','Token result: ','\n', tokens_result,'\n')

    #stremming
    print("Stremming Words:",'\n')
    stemmer = PorterStemmer()
    for word in tokens_result:
        print(stemmer.stem(word))

    #Lemmatization
    print("Lemmatization Words:",'\n')
    lemmatizer=WordNetLemmatizer()
    for word in tokens_result:
        print(lemmatizer.lemmatize(word))


#(2) Sentiment Intensity Analyzer
from nltk.sentiment import SentimentIntensityAnalyzer
with open('Data Text.txt') as f:
    contents = f.readlines()
for sentence in contents:
    s=SentimentIntensityAnalyzer()
    vs=s.polarity_scores(sentence)
    print('\n',sentence,'\n', str(vs))


#(3)Sentiment Classification (Bonus)

import pandas as pd
#from sklearn.model_selection import train_test_split
test=[
      ("This is the best sentiment analysis tool ever!!!",'Postive'),
      ("Thanks for great expreience , am so happy",'Postive'),
      ("the food is great.",'Postive'),
      ("your life is disgusting ",'Negative'),
      ("this is my worst performance.",'Postive'),
      ("I do not want to live anymore",'Negative'),
      ("what you have learned yours and only yours what you want teach different focus the goal not the wrapping paper buddhism can passed others without word about the buddha ",'Neutral'), ]


Sentiment_data=pd.read_csv("Reddit data.csv" ,encoding='unicode_escape')
lst1=Sentiment_data["clean_comment"]
lst2=Sentiment_data["Classify"]
train = list(zip(lst1,lst2))
print(train)
#train,test = train_test_split(data_tuple)
from textblob.classifiers import NaiveBayesClassifier
c1=NaiveBayesClassifier(train)
print(c1.classify("I do not want to live anymore"))
print(c1.accuracy(test))
