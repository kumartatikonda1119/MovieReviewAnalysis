import pandas as pd
data=pd.read_csv('imdb.csv')    
x=data['review']
y=data['sentiment']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english')
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,y_pred)) 
statement = "Lots of lapses from the beginning to the end from the director. Even newcomers in the direction field don't have these kind of mistakes throughout a movie nowadays. Considering everything I would've given just 1 star but, there is still a commendable job done by the actors, so for that it's just a ok movie for a weekend night....P.S. Just to put things into perspectives, no magic as been done as such Waltair Veerayya which is by far the best movie of Bobby. Hoping it's not the last by any means. Good luck Bobby, we all have hope in you. One final thing, heartfully loved your Cameo in."
statement=cv.transform([statement])
print(model.predict(statement))
