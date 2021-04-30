# Importing the required packages
import streamlit as st
import tensorflow as tf
import pandas as pd
import os,urllib
import numpy as np    
import logging
logging.getLogger('tensorflow').disabled = True
from func import *
import joblib


from PIL import Image
import time
from gensim.models import FastText

ft_model = FastText.load("fasttext.model")
def main():
    st.title("Customer Satisfaction Prediction")
    st.subheader(" Predicting Customer Reviews Whether It is Positive Or Negative")
    st.markdown("""
    	#### Description
    	+ The objective is to predict the customer satisfaction (positive or negative) for the purchase made from the brazilain e-commerce site Olist.

    
    	""")
    st.markdown("""
    	#### Steps
        step 1- Give Input Raw Data in .csv.
        step-2 - Choose Unique Customer ID. 
        step-3 - Choose Order IDs of the selected customer. """)  

    # Your code goes below
    # Our Dataset
    st.markdown("""
    	#### Input Raw Data(in .csv)

        """)
   
    @st.cache(allow_output_mutation=True)

    def tfidfWord2Vector(text,ft_words,tfidf_words,tf_values):
        # average Word2Vec
        # compute average word2vec for each review.
        tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
        for sentence in tqdm(text): # for each review/sentence
            vector = np.zeros(300) # as word vectors are of zero length
            tf_idf_weight =0; # num of words with a valid vector in the sentence/review
            for word in sentence.split(): # for each word in a review/sentence
                if (word in ft_words) and (word in tfidf_words):
                    vec = ft_model.wv[word] # embeddings[word] 
                    # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
                    tf_idf = tf_values[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
                    vector += (vec * tf_idf) # calculating tfidf weighted w2v
                    tf_idf_weight += tf_idf
            if tf_idf_weight != 0:
                vector /= tf_idf_weight
            tfidf_w2v_vectors.append(vector)
        tfidf_w2v_vectors = np.asarray(tfidf_w2v_vectors)
        
        return tfidf_w2v_vectors

    @st.cache(allow_output_mutation=True)
    def preprocessing(X):
        X = replace_nan(X)

    #dedublication
        X,index = dedublicate(X)
    
        timestamp_col = ['order_purchase_timestamp','order_approved_at','order_delivered_customer_date',
                            'order_estimated_delivery_date']
        X[timestamp_col]= X[timestamp_col].apply(pd.to_datetime)
        #adding new features 
        X = feat_engg(X)
        #rfm_lvl = rfm_level()
        rfm = rfm_feat(X)
        X = X.merge(rfm ,on ='customer_unique_id',how='left')    

        #dropping columns     
        col= ['order_id','customer_id','order_purchase_timestamp','order_approved_at',
        'order_estimated_delivery_date','customer_unique_id','order_item_id','product_id','seller_id','shipping_limit_date','f_quartile','r_quartile',
        'm_quartile','RFM_Score','RFM_Score_s','product_category_name']
        X.drop(columns=col,axis=1,inplace=True)

        #text preprocessing
        process_txt  = preprocess_text(X['review_comment_message'])
        X['review_comment_message'] = process_txt
        
        #TEXT featurization


        # encoding review comment message using Tfidf weighted W2V
        tfidf = TfidfVectorizer()
        tfidf.fit(X['review_comment_message'])
        
        # we are converting a dictionary with word as a key, and the idf as a value
        tf_values = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
        tfidf_words = set(tfidf.get_feature_names())
        ft_words = list(ft_model.wv.vocab.keys()) # list(embeddings.keys())
        tfidf_w2v_vectors_X = tfidfWord2Vector(X['review_comment_message'].values,ft_words,tfidf_words,tf_values)

        cat_col = ['order_status','payment_type','customer_state','product_category_name_english','RFM_Level']
        X_train = pickle.load(open('X_train.pkl','rb')) 
        cat_feat = cat_feats(cat_col,X_train,X)

        # numerical features
        num=['payment_sequential','payment_installments','payment_value','customer_zip_code_prefix','price',
        'freight_value','product_name_lenght','product_description_lenght','product_photos_qty',
        'product_weight_g','product_length_cm','product_height_cm','product_width_cm',
        'recency','frequency','monetary','sellers_count','products_count','est_delivery_t',
        'act_delivery_t','diff_in_delivery_t','on_time_delivery','avg_prdt_value','total_order_cost',
        'order_freight_ratio','purchase_dayofweek','is_reviewed','words_per_review','day_to_delivery']
        
        X_num = num_feats(X,num)
        
        #tokenization and pad_sequencing 
        textX = create_tokenizer(X_train['review_comment_message'],X['review_comment_message'])

        trainX_os = create_tokenizer(X_train['order_status'],X['order_status'])
        trainX_pt = create_tokenizer(X_train['payment_type'],X['payment_type'])
        trainX_st = create_tokenizer(X_train['customer_state'],X['customer_state'])
        trainX_pc = create_tokenizer(X_train['product_category_name_english'],X['product_category_name_english'])
        trainX_rfm = create_tokenizer(X_train['RFM_Level'],X['RFM_Level'])
        x_train=[textX, trainX_os, trainX_pt, trainX_st, trainX_pc, trainX_rfm, X_num]
        return tfidf_w2v_vectors_X,cat_feat,X_num,textX,x_train,index

    @st.cache(allow_output_mutation=True)
    def preprocessing_1(X):

        #replacing null values
        X = replace_nan(X)


        timestamp_col = ['order_purchase_timestamp','order_approved_at','order_delivered_customer_date',
                            'order_estimated_delivery_date']
        X_= dty_time(X,timestamp_col)
        #X_=X_.drop(['order_delivered_carrier_date'],axis=1)
        #adding new features 
        X_ = feat_engg(X_)
        #rfm_lvl = rfm_level()
        rfm = pickle.load(open('rfm.pkl','rb')) 
        X_ = X_.merge(rfm ,on ='customer_unique_id',how='left')    

        #dropping columns     
        col= ['order_id','customer_id','order_purchase_timestamp','order_approved_at','order_delivered_customer_date',
        'order_estimated_delivery_date','customer_unique_id','order_item_id','product_id','seller_id','shipping_limit_date','f_quartile','r_quartile',
        'm_quartile','RFM_Score','RFM_Score_s','product_category_name']
        X_.drop(columns=col,axis=1,inplace=True)

        #text preprocessing
        process_txt  = preprocess_text(X_['review_comment_message'])
        X_['review_comment_message'] = process_txt

        # encoding review comment message using Tfidf weighted W2V
        tfidf = TfidfVectorizer()
        tfidf.fit(X_['review_comment_message'])

        # we are converting a dictionary with word as a key, and the idf as a value
        tf_values = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
        tfidf_words = set(tfidf.get_feature_names())
        ft_words = list(ft_model.wv.vocab.keys()) # list(embeddings.keys())
        tfidf_w2v_vectors_X = tfidfWord2Vector(X_['review_comment_message'].values,ft_words,tfidf_words,tf_values)

        cat_col = ['order_status','payment_type','customer_state','product_category_name_english','RFM_Level']
        X_train = pickle.load(open('X_train.pkl','rb')) 
        cat_feat = cat_feats(cat_col,X_train,X_)

        # numerical features
        num=['payment_sequential','payment_installments','payment_value','customer_zip_code_prefix','price',
        'freight_value','product_name_lenght','product_description_lenght','product_photos_qty',
        'product_weight_g','product_length_cm','product_height_cm','product_width_cm',
        'recency','frequency','monetary','sellers_count','products_count','est_delivery_t',
        'act_delivery_t','diff_in_delivery_t','on_time_delivery','avg_prdt_value','total_order_cost',
        'order_freight_ratio','purchase_dayofweek','is_reviewed','words_per_review','day_to_delivery']
            
        X_num = num_feats(X_,num)

        #tokenization and pad_sequencing 
        textX = create_tokenizer(X_train['review_comment_message'],X_['review_comment_message'])

        trainX_os = create_tokenizer(X_train['order_status'],X_['order_status'])
        trainX_pt = create_tokenizer(X_train['payment_type'],X_['payment_type'])
        trainX_st = create_tokenizer(X_train['customer_state'],X_['customer_state'])
        trainX_pc = create_tokenizer(X_train['product_category_name_english'],X_['product_category_name_english'])
        trainX_rfm = create_tokenizer(X_train['RFM_Level'],X_['RFM_Level'])
        x_train=[textX, trainX_os, trainX_pt, trainX_st, trainX_pc, trainX_rfm, X_num]
        return tfidf_w2v_vectors_X,cat_feat,X_num,textX,x_train
   

    @st.cache(allow_output_mutation=True)
    def function_1(X):

        if len(X)==1:
            tfidf_w2v_vectors_X,cat_feat,X_num,textX,x_train = preprocessing_1(X)
        else:
            tfidf_w2v_vectors_X,cat_feat,X_num,textX,x_train,index = preprocessing(X)

        X_tr = hstack((tfidf_w2v_vectors_X,list(cat_feat.values())[0],list(cat_feat.values())[1],list(cat_feat.values())[2],
                            list(cat_feat.values())[3],list(cat_feat.values())[4],X_num)).tocsr()
            # load the model from file
        encoder = load_model('encoder.h5')

        # encode the train data
        X_encode = encoder.predict(X_tr)
        X_encode_1 = X_encode.reshape(X_encode.shape[0],X_encode.shape[1],1)


        # merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039

        X_tr_num = hstack((list(cat_feat.values())[0],list(cat_feat.values())[1],list(cat_feat.values())[2],
                        list(cat_feat.values())[3],list(cat_feat.values())[4],X_num)).tocsr()

        x_tr_num = np.array(X_tr_num.todense()).reshape(X_tr_num.shape[0],X_tr_num.shape[1],1)
            
        #loading models
        model_1 = load_model('model_1.h5')
        model_2 = load_model('model_2.h5')
        model_3 = load_model('model_3.h5')
        model_4 = load_model('model_4.h5')
        model_5 = load_model('model_5.h5')

        #saving the model
        filename = 'stacknn2.model'
        stacknn2 = joblib.load(open(filename,'rb'))

        
        #prediction of train data
        y_pred1 = model_1.predict(X_encode)
        y_pred2 = model_2.predict(X_encode_1)
        y_pred3 = model_3.predict(x_train)
        y_pred4 = model_4.predict([textX,x_tr_num])
        y_pred5 = model_5.predict([textX,X_tr_num])
        

        y_pred = stacknn2.predict(np.stack((np.greater(y_pred1,0.5).astype(int)[:,0],
                                                    np.greater(y_pred2,0.5).astype(int)[:,0],
                                                    np.greater(y_pred3,0.5).astype(int)[:,0],
                                                    np.greater(y_pred4,0.5).astype(int)[:,0],
                                                    np.greater(y_pred5,0.5).astype(int)[:,0]),axis=-1))
        if len(X)==1:
            return y_pred
        else:
            return y_pred,index
         # To Improve speed and cache data

    @st.cache(allow_output_mutation=True)

    def explore_data(file):
        df = pd.read_csv(file) 
        df=df.drop(["Unnamed: 0"],axis=1)
        return df 
        

    uploaded_file = st.file_uploader("input data", type=".csv")
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)


     
    
    
        st.dataframe(explore_data(uploaded_file))  # Same as st.write(df) 

        df = explore_data(uploaded_file)
        if len(df)==1:
            pred = function_1(explore_data(uploaded_file))
            X_new = df.reset_index()
            X_new['y_pred'] = list(pred)
        else:
            pred,index_=function_1(explore_data(uploaded_file))
            X_new = df.drop(index=index_,axis=0)
            X_new = X_new.reset_index()
            X_new['y_pred'] = list(pred)

        customer_id = list(explore_data(uploaded_file).customer_unique_id.unique())

        options_1 = st.selectbox( 'Select Customer Unique ID', customer_id)
        st.write('You selected:', options_1)
    

        
        if options_1 is not None:

            options = st.selectbox( 'Select Order ID', list(explore_data(uploaded_file)[explore_data(uploaded_file)['customer_unique_id']== options_1].order_id))
            st.write('You selected:', options)

            if options is not None:



                a=list(X_new[(X_new["customer_unique_id"]== options_1 ) & (X_new["order_id"]== options )]["y_pred"])

                st.markdown("""
                #### Prediction:

                """)

                img = Image.open("neg-removebg-preview.png")
                img_1 = Image.open("pos-removebg-preview.png") 

                if a[0]==1:
                    st.image(img_1)
                else:
                    st.image(img)




if __name__ == "__main__":
    main()        


