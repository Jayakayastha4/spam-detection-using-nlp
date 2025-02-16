
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import  MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
# dataset is downloaded from kaggle
df=pd.read_csv(r"C:\Users\HP\Downloads\python vs code\nlpfor spam\mail_data.csv")
#print(df.head())
# Count the occurrences of each class
ham_count = df[df['Category'] == 'ham'].shape[0]
spam_count = df[df['Category'] == 'spam'].shape[0]

# Print the ratio
print(f'Ham messages: {ham_count}, Spam messages: {spam_count}')
print(f'Ratio (Ham:Spam) = {ham_count / spam_count:.2f}:1')
# checking the distribution of ham and spam message
df['Category'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], figsize=(6, 4))

plt.xlabel('Message Type')
plt.ylabel('Count')
plt.title('Distribution of Ham and Spam Messages')
plt.xticks(rotation=0)  # Keep labels horizontal

plt.show()
#print(f'is there any null value in dataset:{df.isnull().sum()}')
#print(f'duplicate value in dataset:{df.duplicated().sum()}')
df.drop_duplicates(inplace=True)
#print(f'duplicate value in dataset:{df.duplicated().sum()}')
#converting messages to lower case because it Reduces Vocabulary Size,Improves Text Matching & Tokenization,Better Feature Extraction
df['Message']=df['Message'].str.lower()
#print(df.head())
pattern = r"[@#^]|https?:\/\/.*[\r\n]*"

# Apply regex replacement
df['Message'] = df['Message'].str.replace(pattern, "", regex=True)
import string
# removes all punctuation from the text in the clean_message column and stores the cleaned text in the Message column.
df['Message']=df['Message'].str.translate(str.maketrans('','',string.punctuation))
stop_words=stopwords.words('english')
# removing stop words from the 'Message' column
df['Message']=df['Message'].apply(lambda x :' '.join([word for word in x.split()if word not in (stop_words)]))
#print(df.head())
#sentence tokenization to preserve sentence structure,Improves Downstream NLP Tasks such assentiment analysis, named entity recognition (NER)
from nltk.tokenize import sent_tokenize
df['tokenize_text']=df['Message'].apply(sent_tokenize)
#print(df.head())
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
# we apply steemere to Reduce words to their root form such as "running" → "run",Improve text normalization,Reduce vocabulary size
stemmere=PorterStemmer()
def stem_words(text):
    return " ".join([stemmere.stem(word) for word in text.split()])
df['stem_msg']=df['Message'].apply(stem_words)
#print(df.head())
cv=CountVectorizer()
X=cv.fit_transform(df['stem_msg']).toarray()
#print(X.shape)
y=df['Category']
#print(y.shape)
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
Y=LE.fit_transform(y)
X_train,X_test,Ytrain,Ytest=train_test_split(X,Y,train_size=0.8,random_state=42)

def evaluate(Ytest,y_pred):
    acc=accuracy_score(Ytest,y_pred)
    pre=precision_score(Ytest,y_pred)
    conm=confusion_matrix(Ytest,y_pred)
    return acc,pre,conm

svc=SVC(kernel='sigmoid',gamma=1.0)
svc.fit(X_train,Ytrain)
svc_pred=svc.predict(X_test)
accuracy_svc,precision_svc,confusion_svc=evaluate(Ytest,svc_pred)

knc=KNeighborsClassifier()
knc.fit(X_train,Ytrain)
knc_pred=knc.predict(X_test)
accuracy_knc,precision_knc,confusion_knc=evaluate(Ytest,knc_pred)

mnb=MultinomialNB()
mnb.fit(X_train,Ytrain)
mnb_pred=mnb.predict(X_test)
accuracy_mnb,precision_mnb,confusion_mnb=evaluate(Ytest,mnb_pred)

dtc=DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train,Ytrain)
dtc_pred=dtc.predict(X_test)
accuracy_dtc,precision_dtc,confusion_dtc=evaluate(Ytest,dtc_pred)

lrc=LogisticRegression(class_weight='balanced', random_state=42)
lrc.fit(X_train,Ytrain)
lrc_pred=lrc.predict(X_test)
accuracy_lrc,precision_lrc,confusion_lrc=evaluate(Ytest,lrc_pred)

rfc=RandomForestClassifier(n_estimators=50,random_state=2)
rfc.fit(X_train,Ytrain)
rfc_pred=rfc.predict(X_test)
accuracy_rfc,precision_rfc,confusion_rfc=evaluate(Ytest,rfc_pred)

abc=AdaBoostClassifier(n_estimators=50,random_state=2)
abc.fit(X_train,Ytrain)
abc_pred=abc.predict(X_test)
accuracy_abc,precision_abc,confusion_abc=evaluate(Ytest,abc_pred)

etc=ExtraTreesClassifier(n_estimators=50,random_state=2)
etc.fit(X_train,Ytrain)
etc_pred=etc.predict(X_test)
accuracy_etc,precision_etc,confusion_etc=evaluate(Ytest,etc_pred)

xgb=XGBClassifier()
xgb.fit(X_train,Ytrain)
xgb_pred=xgb.predict(X_test)
accuracy_xgb,precision_xgb,confusion_xgb=evaluate(Ytest,xgb_pred)
evaluation_data = {
    'Model': ['SVC', 'KNN', 'MultinomialNB', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'AdaBoost', 'Extra Tree', 'XGBoost'],
    'Accuracy': [accuracy_svc, accuracy_knc, accuracy_mnb, accuracy_dtc, accuracy_lrc, accuracy_rfc, accuracy_abc, accuracy_etc, accuracy_xgb],
    'Precision': [precision_svc, precision_knc, precision_mnb, precision_dtc, precision_lrc, precision_rfc, precision_abc, precision_etc, precision_xgb]
}

# Create a dataframe
evaluation_df = pd.DataFrame(evaluation_data)

# Sort the dataframe based on Accuracy and Precision columns in descending order
evaluation_df = evaluation_df.sort_values(by=['Accuracy', 'Precision'], ascending=False)

print(evaluation_df)

print(f"from evaluation df we found logistic regression have better performance with accuracy:{accuracy_lrc},precision:{precision_lrc} ")


import matplotlib.pyplot as plt
import numpy as np
"""
# Define the models and their accuracies and precisions
models = ['SVC', 'KNN', 'MultinomialNB', 'Decision Tree', 'Logistic Regression', 
          'Random Forest', 'AdaBoost', 'Extra Tree', 'XGBoost']

accuracies = [accuracy_svc, accuracy_knc, accuracy_mnb, accuracy_dtc, 
              accuracy_lrc, accuracy_rfc, accuracy_abc, accuracy_etc, accuracy_xgb]

precisions = [precision_svc, precision_knc, precision_mnb, precision_dtc, 
              precision_lrc, precision_rfc, precision_abc, precision_etc, precision_xgb]

# Set bar width
bar_width = 0.4
x = np.arange(len(models))

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot accuracy and precision bars side by side
ax.bar(x - bar_width/2, accuracies, bar_width, label='Accuracy', color='skyblue')
ax.bar(x + bar_width/2, precisions, bar_width, label='Precision', color='salmon')

# Set labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Accuracy and Precision of Different Models')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")

# Add legend
ax.legend()

# Show plot
#plt.tight_layout()
#plt.show()
"""