import pandas as pd
from transformers import BertTokenizer, TFBertForPreTraining, TFBertModel,TFBertForSequenceClassification
import numpy as np
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from scipy.special import softmax
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import os
import datetime
from torch.autograd import Variable

class News_Features():
    
    """
    A class used for feature extarctiion from news

    ...

    Attributes
    ----------
    tickers_path : str
        path for the tickers csv file
        
    tickers : array
        array of tickers
         
    save_path : str
        path to the directory of news features data
        
    news_path : str
        path to the directory of news news data

    model_path : str
        path to finBERT directory
    
    model : torch model
        finBERT model
        
    tokenize : function
        used to tokenize text
        
    device : torch device
        
    Methods
    -------
    define_model()
        defines the model, device and tokenizer
        
    pool(array)
        smoothing of sentiments
        
    sentiment_estimation(row)
        extracts sentiments from article
    
    extract_features(df)
        extract features from articles 
        
    news_engineering(ticker)
        extract features from articles of ticker and saves them
        
    news_engineering_all()
        extract features from articles of all ticker and saves them
        
    """
    
    
    
    def __init__(self,tickers_path='Input/tickers.csv',save_path='Output/News/NewsFeatures/',news_path='Output/News/',model_path='../models/FinancialPhraseBank/'):
        
        """
        Parameters
        ----------
        tickers_path : str
            path for the tickers csv file
            
        save_path : str
            path to the directory of news data
            
        news_path : str
            path to the directory of news news data

        model_path : str
            path to finBERT directory
        """
        
        self.tickers_path=tickers_path
        self.save_path=save_path
        self.news_path=news_path
        self.model_path=model_path
        
        self.model=None
        self.tokenize=None
        self.device=None
        
        self.tickers=pd.read_csv(tickers_path).iloc[:,0]
        
        self.define_model()
        
        
    def define_model(self):
        
        
        """defines the model, device and tokenizer

        Parameters
        ----------

            
        """
        
        max_len=64
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize(sentence):
            return tokenizer.encode_plus(
                            sentence,                      
                            add_special_tokens = True, # add [CLS], [SEP]
                            max_length = max_len, # max length of the text that can go to BERT
                            pad_to_max_length = True, # add [PAD] tokens
                            return_attention_mask = True, # add attention mask to not focus on pad tokens
                      )

        model = BertForSequenceClassification.from_pretrained(self.model_path,num_labels=3,cache_dir=None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device) 
        model.eval()
        model.cuda()
        
        self.model=model
        self.tokenize=tokenize
    
    
    def pool(self,array):
        
        """smoothing of sentiments

        Parameters
        ----------
        array : array
            array of sentiments (shape=(N_sentences,sentiments))
            

        Returns
        -------
        array
            smoothed array
        """
        
        
        
        pos=array[:,0]
        neg=array[:,1]
        neu=array[:,2]

        L=[pos,neg,neu]

        for i in range(len(L)):
            arr_pad=np.zeros(len(L[i])+4)
            arr_pad[2:len(L[i])+2]=L[i]
            L[i]=[np.average([arr_pad[i-2],arr_pad[i-1],arr_pad[i],arr_pad[i+1],arr_pad[i+2]],
                            weights=[0.2,0.6,1,0.6,0.2]) for i in range(2,len(arr_pad)-2)]


        array=np.transpose(np.vstack([L[0],L[1],L[2]]))

        return array
    
        
    def sentiment_estimation(self,row):
        
        
        """extracts sentiments from article

        Parameters
        ----------
        row : row of dataframe
            data of an article
            

        Returns
        -------
        Series
            [pos,neg,neutral]
        """
    
        hed=row.hed
        led=row.led
        text=row.text


        input_ids=[]
        token_type_ids=[]
        attention_mask=[]

        
        for sentence in [hed,led]+sent_tokenize(text):
            if (str(sentence)=='nan'):
                continue
            tokens=self.tokenize(sentence)
            input_ids.append(tokens['input_ids'])
            token_type_ids.append(tokens['token_type_ids'])
            attention_mask.append(tokens['attention_mask']) 


        input_ids= Variable(torch.tensor(input_ids, dtype=torch.long).to(self.device)).cuda()
        token_type_ids= Variable(torch.tensor(token_type_ids, dtype=torch.long).to(self.device)).cuda()
        attention_mask= Variable(torch.tensor(attention_mask, dtype=torch.long).to(self.device)).cuda()


        with torch.no_grad():
            y_pred=self.model(input_ids,token_type_ids,attention_mask).cpu().numpy()
        
        y_pred=self.pool(y_pred)
        
        y_pred=np.mean(y_pred,axis=0)
        y_pred = softmax(y_pred, axis=0)

        return pd.Series(y_pred)
    
    
    def extract_features(self,df):
        
        """extract features from articles

        Parameters
        ----------
        df : dataframe
            dataframe of articles
            

        Returns
        -------
        dataframe
            dataframe of articles with additional features
        """
        
        if(len(df)!=0):
            df.index=pd.to_datetime(df.published_at)
            df.text=df.text.apply(lambda x: ' '.join(map(str, eval(x))))
            df['sentiment_positive'] = 0
            df['sentiment_negative'] = 0
            df['sentiment_neutral'] = 0
            df[['sentiment_positive','sentiment_negative','sentiment_neutral']]=df.apply(self.sentiment_estimation, axis=1)
            df['len_n2_codes'] = df['n2_codes'].map(lambda x: 0 if x==0 else len(eval(x)))
            df['len_channels'] = df['channels'].map(lambda x: 0 if x==0 else len(eval(x)))
        else:
            df=pd.DataFrame(columns=['art_type', 'source', 'channel_name', 'hed', 'led', 'published_at',
           'updated_at', 'dateline', 'keywords', 'n2_codes', 'slug', 'text',
           'channels', 'sentiment_positive', 'sentiment_negative',
           'sentiment_neutral', 'len_n2_codes', 'len_channels'])
            df_features.index=df_features.published_at

        return df
    
    
    def news_engineering(self,ticker):
        
        """extract features from articles of ticker and saves them
        Parameters
        ----------
        ticker : str
            ticker of as stock (ex: AAPL.O)
            

        Returns
        -------
        dataframe
            dataframe of articles with additional features
        """
        
        if(os.path.exists(self.news_path+ticker+'.csv')):
            df_news=pd.read_csv(self.news_path+ticker+'.csv')
            if (len(df_news)==0):
                #No news
                df_features=pd.DataFrame(columns=['art_type', 'source', 'channel_name', 'hed', 'led', 'published_at',
               'updated_at', 'dateline', 'keywords', 'n2_codes', 'slug', 'text',
               'channels', 'sentiment_positive', 'sentiment_negative',
               'sentiment_neutral', 'len_n2_codes', 'len_channels'])
                df_features.index=df_features.published_at
            
            else:
                if (os.path.exists(self.save_path+ticker+'.csv')):
                    df_features=pd.read_csv(self.save_path+ticker+'.csv',index_col='published_at')
                    df_features.index=pd.to_datetime(df_features.index)
                    if(len(df_features)!=0):
                        last_news_feat=df_features.index[0]
                        last_news=pd.to_datetime(df_news.published_at)[0]
                        if(last_news>last_news_feat):
                            df_news=df_news[pd.to_datetime(df_news.published_at)>last_news_feat]
                            df_features_new=self.extract_features(df_news)
                            df_features=pd.concat([df_features_new,df_features])
                    else:
                        df_features=self.extract_features(df_news)             
                else:
                    df_features=self.extract_features(df_news)

        else:
            #No news
            df_features=pd.DataFrame(columns=['art_type', 'source', 'channel_name', 'hed', 'led', 'published_at',
           'updated_at', 'dateline', 'keywords', 'n2_codes', 'slug', 'text',
           'channels', 'sentiment_positive', 'sentiment_negative',
           'sentiment_neutral', 'len_n2_codes', 'len_channels'])
            df_features.index=df_features.published_at

        df_features.to_csv(self.save_path+ticker+'.csv')
        print('saved '+self.save_path+ticker+'.csv')

        return df_features
    
    def news_engineering_all(self):
        
        
        """extract features from articles of all tickers and saves them
        

        Returns
        -------
        list
            list of dataframes of articles with additional features
        """
        
        dff=[]
        for ticker in self.tickers:
            df=self.news_engineering(ticker)
            dff.append(df)

        return dff
    
    
