#Importing Liraries

# Importing Libraries
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import shutil
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use(u'nbAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from scipy.sparse import hstack
from wordcloud import WordCloud


# Utilities
#from viz_utils import *
#from custom_transformers import *
#from ml_utils import *

# DataPrep
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
# Modelin
#Metrics
from sklearn.metrics import log_loss,accuracy_score, confusion_matrix, f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#Importing the Libraries

from tensorflow.keras.models import load_model
#text data preprocessing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
nltk.download('rslp')



def replace_nan(df):

  df["order_approved_at"].fillna(df["order_purchase_timestamp"], inplace=True)
  df["order_delivered_customer_date"].fillna(df["order_estimated_delivery_date"], inplace=True)

  #dropping order delivery carrier date
  #df.drop(labels='order_delivered_carrier_date',axis=1,inplace=True)
  # Handling missing values of numerical features
  df['product_weight_g'].fillna(df['product_weight_g'].median(),inplace=True)
  df['product_length_cm'].fillna(df['product_length_cm'].median(),inplace=True)
  df['product_height_cm'].fillna(df['product_height_cm'].median(),inplace=True)
  df['product_width_cm'].fillna(df['product_width_cm'].median(),inplace=True)
  #Handling missing values of text column

  # filling null value of review comments with no_review
  df['review_comment_message'].fillna('nao_reveja',inplace=True)
  return df


def dedublicate(df):
    #Deduplication of entries
    index_1 = list(df.index)
    df= df.drop_duplicates(subset={'order_id','customer_id','order_purchase_timestamp','order_delivered_customer_date'}, keep='first', inplace=False)
    index_2 = list(df.index)
    for i in index_2:
      index_1.remove(i)
    df=df.reindex()
    return df,index_1

def feat_engg(data_n):

    # calculating estimated delivery time
    data_n['est_delivery_t'] = (data_n['order_estimated_delivery_date'] - data_n['order_approved_at']).dt.days

    # calculating actual delivery time
    data_n['act_delivery_t'] = (data_n['order_delivered_customer_date'] - data_n['order_approved_at']).dt.days

    # calculating diff_in_delivery_time
    data_n['diff_in_delivery_t'] = data_n['est_delivery_t'] - data_n['act_delivery_t']

    # finding if delivery was lare
    data_n['on_time_delivery'] = data_n['order_delivered_customer_date'] < data_n['order_estimated_delivery_date']
    data_n['on_time_delivery'] = data_n['on_time_delivery'].astype('int')

    # calculating mean product value
    data_n['avg_prdt_value'] = data_n['price']/data_n['products_count']

    # finding total order cost
    data_n['total_order_cost'] = data_n['price'] + data_n['freight_value']

    # calculating order freight ratio
    data_n['order_freight_ratio'] = data_n['freight_value']/data_n['price']

    # finding the day of week on which order was made
    data_n['purchase_dayofweek'] = pd.to_datetime(data_n['order_purchase_timestamp']).dt.dayofweek

    # adding is_reviewed where 1 is if review comment is given otherwise 0.
    data_n['is_reviewed'] = (data_n['review_comment_message'] != 'no_review').astype('int')
    # Getting the number of words by splitting them by a space
    words_per_review = data_n['review_comment_message'].apply(lambda x: len(str(x).split(" ")))
    data_n['words_per_review'] = words_per_review 
    data_n['day_to_delivery']=(data_n['order_delivered_customer_date']-data_n['order_purchase_timestamp']).dt.days
    return data_n

   # Define rfm_level function
def rfm_level(d):
  if d['RFM_Score_s'] >= 9:
        return 'Can\'t Loose Them'
  elif ((d['RFM_Score_s'] >= 8) and (d['RFM_Score_s'] < 9)):
      return 'Champions'
  elif ((d['RFM_Score_s'] >= 7) and (d['RFM_Score_s'] < 8)):
      return 'Loyal'
  elif ((d['RFM_Score_s'] >= 6) and (d['RFM_Score_s'] < 7)):
      return 'Potential'
  elif ((d['RFM_Score_s'] >= 5) and (d['RFM_Score_s'] < 6)):
      return 'Promising'
  elif ((d['RFM_Score_s'] >= 4) and (d['RFM_Score_s'] < 5)):
      return 'Needs Attention'
  else:
      return 'Require Activation'

def rfm_feat(df):
  PRESENT = datetime(2018,9,3)
  rfm= df.groupby('customer_unique_id').agg({'order_purchase_timestamp': lambda date: (PRESENT - date.max()).days,
                                        'order_id': lambda num: len(num),
                                        'payment_value': lambda price: price.sum()})
  rfm.columns=['recency','frequency','monetary']
  rfm['recency'] = rfm['recency'].astype(int)
  rfm['frequency'] = rfm['frequency'].astype(int)
  rfm['monetary'] = rfm['monetary'].astype(float)
  # Create labels for Recency and Frequency
  def partition(x):
    if x < 10:
      return 1
    if 10<=x<=35:
      return 2
    if 35<x<=50:
      return 3
    if 50<x<=75:
      return 4 
  rfm['f_quartile']=rfm['frequency'].map(lambda cw : partition(cw) ) 
    
    # checking the review score now
  rfm.f_quartile.value_counts()
  r_labels = range(4, 0, -1)
  m_labels= range(1,5)

  rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, r_labels)
  rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, m_labels)
  rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
  rfm['RFM_Score_s'] = rfm[['r_quartile','f_quartile','m_quartile']].sum(axis=1)
  #rfm_level = pickle.load(open('data_pipeline/rfm_level.pkl','rb'))
  rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)
  return rfm  

def cat_feats(cat_col,X_train,X):
  keys,values=[],[]
  for i in cat_col:
    vectorizer = CountVectorizer(binary= True)
    vectorizer.fit(X_train[i].values)
    a = 'feat_' + i[0:3]
    b = vectorizer.transform(X[i].values)
    keys.append(a)
    values.append(b)
  cat_feat = dict(zip(keys, values))
  return cat_feat

def num_feats(X,num):

  tr=[]
  for col in num: 
    normalizer = Normalizer()
    normalizer.fit(X[col].values.reshape(-1,1))
    X_norm = normalizer.transform(X[col].values.reshape(-1,1))
    tr.append(X_norm)
  X_num = np.hstack((tr))
  return X_num

def preprocess_text(texts):
    #removing 'nao' & 'nem'
    # portugese language stopwords
    stopwords_pt = stopwords.words("portuguese")
    stopwords_pt.remove('não')
    stopwords_pt.remove('nem')
    hyperlinks = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # check for hyperlinks
    dates = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}' # check for dates
    currency_symbols = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+' # check for currency symbols
    preprocessed_text = []
    stemmer = RSLPStemmer() # portugese nltk stemmer
    for sent in tqdm(texts):
        sent = re.sub(r"[\n\t\-\\\/]"," ",sent)#removing the new line,tab  
        sent = re.sub(hyperlinks, ' url ', sent) # replacing hyperlinks with 'url'
        sent = re.sub(dates, ' ', sent) # removing dates
        sent = re.sub(currency_symbols, ' dinheiro ', sent) # replacing currency symbols with 'dinheiro'
        sent = re.sub('[0-9]+', ' numero ', sent) # removing digits
        sent = re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', sent) # replacing no with negative
        sent = re.sub('\W', ' ', sent) # removing extra whitespaces
        sent = re.sub('\s+', ' ', sent) # removing extra spaces
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords_pt) # removing stopwords
        sent = ' '.join(stemmer.stem(e.lower()) for e in sent.split()) # stemming the words
        preprocessed_text.append(sent.lower().strip())
        
    return preprocessed_text



# fit a tokenizer
def create_tokenizer(lines,line_):
  tokenizer =  Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
  tokenizer.fit_on_texts(lines)
  # calculate the maximum document length
  length = max([len(s.split()) for s in lines])
	# integer encode
  encoded = tokenizer.texts_to_sequences(line_)
	# pad encoded sequences
  padded = pad_sequences(encoded, maxlen=length, padding='post')
  return padded

def dty_time(X,timestamp_col):
  x=pd.DataFrame(X)
  for i in timestamp_col:
    x[i]= x[i].astype(dtype='datetime64[ms]')
  return x


