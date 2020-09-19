import time
import datetime
from urllib.request import urlopen
import json
import pandas as pd
import numpy as np
import time
import pickle
import tqdm
import os

class News_fetcher:
    
    
    """
    A class used for news data scrapping from Reuters

    ...

    Attributes
    ----------
    tickers_path : str
        path for the tickers csv file
        
    tickers : array
        array of tickers
        
    lookback : int
        number of lookback months
        
    save_path : str
        path to the directory of news data

    stop_date : datetime
        first date in data 
    
    repeat_times : int
        maximal number of request tries
        
    Methods
    -------
    get_paths(ticker,date=None)
        gets last 20 articles paths starting date and down
        
    get_article_data(path)
        get article from path
        
    get_articles(ticker)
        get articles for ticker from now until stop_date and saves them
    
    get_all_articles()
        get articles for all tickers from now until stop_date and saves them
        
    """
    
    
    
    def __init__(self,tickers_path,save_path,lookback=1):
               
        """
        Parameters
        ----------
        tickers_path : str
            path for the tickers csv file
            
        save_path : str
            path to the directory of news data
            
        lookback : int
            number of lookback months
        """
        
        self.tickers_path=tickers_path
        self.lookback=lookback
        self.tickers=pd.read_csv(tickers_path).iloc[:,0].values
        self.stop_date=(pd.to_datetime(datetime.datetime.today())-np.timedelta64(lookback,'M')).timestamp()
        self.repeat_times=3
        self.save_path=save_path
    
    def get_paths(self,ticker,date=None):
        
        
        """gets last 20 articles paths starting date and down

        Parameters
        ----------
        ticker : str
            ticker (ex: AAPL.O)
            
        date : int
            date in epoch time
            

        Returns
        -------
        list
            list of article paths
        """
        
        if(date==None):
            response = urlopen('https://wireapi.reuters.com/v8/feed/rcom/us/marketnews/ric:'+ticker)
        else:
            date=str(int(date))
            response = urlopen('https://wireapi.reuters.com/v8/feed/rcom/us/marketnews/ric:'+ticker+'?until='+date)
            
        data = response.read().decode('utf-8')
        data = json.loads(data)

        art_paths=[]
        for art_index in range(len(data['wireitems'])):
            try :
                art_paths.append(data['wireitems'][art_index]['templates'][0]['template_action']['api_path'])
            except:
                continue
                
        return art_paths
    
    def get_article_data(self,path):
        
        """get article from path

        Parameters
        ----------
        path : str
            API path to the article
            

        Returns
        -------
        list
            list of article data
        """

        article = urlopen('https://wireapi.reuters.com/v8'+path)
        article = article.read().decode('utf-8')
        article = json.loads(article)

        art_type=article['wireitems'][0]['templates'][0]['type']

        source=article['wireitems'][0]['templates'][0]['story']['source']

        channel_name=article['wireitems'][0]['templates'][0]['story']['channel']['name']

        hed=article['wireitems'][0]['templates'][0]['story']['hed']

        led=article['wireitems'][0]['templates'][0]['story']['lede']

        published_at=article['wireitems'][0]['templates'][0]['story']['published_at']

        updated_at=article['wireitems'][0]['templates'][0]['story']['updated_at']

        dateline=article['wireitems'][0]['templates'][0]['story']['dateline']

        keywords=article['wireitems'][0]['templates'][0]['story']['keywords']

        n2_codes=article['wireitems'][0]['templates'][0]['story']['n2_codes']

        slug=article['wireitems'][0]['templates'][0]['story']['slug']

        text=[]
        for part in article['wireitems'][0]['templates'][0]['story']['body_items']:
            try:
                text.append(part['content'])
            except:
                continue

        channels=article['wireitems'][0]['templates'][0]['story']['channel_names']

        return[art_type,source,channel_name,hed,
                led,published_at,updated_at,dateline,keywords,n2_codes,slug,text,channels]
    
    def get_articles(self,ticker):
        
        """get articles for ticker from now until stop_date and saves them

        Parameters
        ----------
        ticker : str
            ex: AAPL.O
            

        Returns
        -------
        dataframe
            dataframe of articles data
        """
        
        df=pd.DataFrame(columns=['art_type','source','channel_name','hed',
                         'led','published_at','updated_at','dateline','keywords','n2_codes','slug','text','channels'])
        
        save_path=self.save_path+ticker+'.csv'
        
        #If the ticker does not exist in our data
        if (not os.path.exists(save_path)):
            #init
            art_paths=None
            done=False
            for _ in range(self.repeat_times): # repeat in case of http failure
                try:
                    time.sleep(np.random.poisson(3))
                    art_paths=self.get_paths(ticker)
                    break
                except:
                    continue
            
            if(art_paths!=None):      
                for path in art_paths:
                    try:
                        time.sleep(np.random.poisson(3))
                        df.loc[len(df)]=self.get_article_data(path)
                        print('fetching '+ticker+' '+str(df.iloc[-1,6]))
                    except:
                        continue
            else:
                done=True
                
            df.to_csv(save_path,index=False)
            
            
            while(done==False and datetime.datetime.strptime(df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp() > self.stop_date):
                art_paths=None
                for _ in range(self.repeat_times): # repeat in case of http failure
                    try:
                        time.sleep(np.random.poisson(3))
                        art_paths=self.get_paths(ticker,datetime.datetime.strptime(df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp()*(10**9))
                        break
                    except:
                        continue
                        
                if(art_paths!=None):
                    
                    for path in art_paths:
                        try:
                            time.sleep(np.random.poisson(3))
                            df.loc[len(df)]=self.get_article_data(path)
                            print('fetching '+ticker+' '+str(df.iloc[-1,6]))
                        except:
                            continue
                else:
                    done=True
                df.to_csv(save_path,index=False)
                        
        #If the ticker does exist in our data
        else:
            df=pd.read_csv(save_path)
            if(len(df)==0):
               #init
                done=False
                art_paths=None
                for _ in range(self.repeat_times): # repeat in case of http failure
                    try:
                        time.sleep(np.random.poisson(3))
                        art_paths=self.get_paths(ticker)
                        break
                    except:
                        continue

                if(art_paths!=None):      
                    for path in art_paths:
                        try:
                            time.sleep(np.random.poisson(3))
                            df.loc[len(df)]=self.get_article_data(path)
                            print('fetching '+ticker+' '+str(df.iloc[-1,6]))
                        except:
                            continue
                else:
                    done=True
                    
                    
                df.to_csv(save_path,index=False)

                
                while(done==False and datetime.datetime.strptime(df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp() > self.stop_date):
                    art_paths=None
                    for _ in range(self.repeat_times): # repeat in case of http failure
                        try:
                            time.sleep(np.random.poisson(3))
                            art_paths=self.get_paths(ticker,datetime.datetime.strptime(df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp()*(10**9))
                            break
                        except:
                            continue
                    if(art_paths!=None):
                        for path in art_paths:
                            try:
                                time.sleep(np.random.poisson(3))
                                df.loc[len(df)]=self.get_article_data(path)
                                print('fetching '+ticker+' '+str(df.iloc[-1,6]))
                                
                            except:
                                continue
                    else:
                        done=True
                    df.to_csv(save_path,index=False)
               
               
               
               
            else:
                #extract last_pub_date and start_pub_date
                last_pub_date=datetime.datetime.strptime(df.iloc[0,6], "%Y-%m-%dT%H:%M:%SZ").timestamp() #max
                first_pub_date=datetime.datetime.strptime(df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp() #min


                if (first_pub_date > self.stop_date):
                    #We need to add more articles (end)
                    done=False
                    while(done==False and datetime.datetime.strptime(df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp() > self.stop_date):
                        art_paths=None
                        for _ in range(self.repeat_times): # repeat in case of http failure
                            try:
                                time.sleep(np.random.poisson(3))
                                art_paths=self.get_paths(ticker,datetime.datetime.strptime(df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp()*(10**9))
                                break
                            except:
                                continue
                        if(art_paths!=None):
                            for path in art_paths:
                                try:
                                    time.sleep(np.random.poisson(3))
                                    df.loc[len(df)]=self.get_article_data(path)
                                    print('fetching '+ticker+' '+str(df.iloc[-1,6]))
                                    
                                except:
                                    continue
                        else:
                            done=True

                        df.to_csv(save_path,index=False)


                if (last_pub_date < pd.to_datetime(datetime.datetime.today()).timestamp()):
                    #we need to add more articles (begining)

                    new_df=pd.DataFrame(columns=['art_type','source','channel_name','hed',
                                     'led','published_at','updated_at','dateline','keywords','n2_codes','slug','text','channels'])

                    art_paths=None
                    for _ in range(self.repeat_times): # repeat in case of http failure
                        try:
                            time.sleep(np.random.poisson(3))
                            art_paths=self.get_paths(ticker)
                            break
                        except:
                            continue
                    if(art_paths!=None):
                        for path in art_paths:
                            try:
                                time.sleep(np.random.poisson(3))
                                new_df.loc[len(new_df)]=self.get_article_data(path)
                                print('fetching '+ticker+' '+str(new_df.iloc[-1,6]))
                                if(datetime.datetime.strptime(new_df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp()<=last_pub_date):
                                    break
                            except:
                                continue
                    df.to_csv(save_path,index=False)

                    done=False
                    while(done==False and datetime.datetime.strptime(new_df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp() > last_pub_date):
                        art_paths=None
                        for _ in range(self.repeat_times): # repeat in case of http failure
                            try:
                                time.sleep(np.random.poisson(3))
                                art_paths=self.get_paths(ticker,datetime.datetime.strptime(new_df.iloc[-1,6], "%Y-%m-%dT%H:%M:%SZ").timestamp()*(10**9))
                                break
                            except:
                                continue
                        if(art_paths!=None):
                            for path in art_paths:
                                try:
                                    time.sleep(np.random.poisson(3))
                                    new_df.loc[len(new_df)]=self.get_article_data(path)
                                    print('fetching '+ticker+' '+str(new_df.iloc[-1,6]))
                                    
                                except:
                                    continue

                        else:
                            done=True

                        df.to_csv(save_path,index=False)


                    df=pd.concat([new_df,df])


        df=df.drop_duplicates(['published_at'])      
        
        df.to_csv(save_path,index=False)
            
        print('saved '+ticker+' in '+save_path)
             
        return df
    
    
    def get_all_articles(self):
        
        """get articles for all tickers from now until stop_date and saves them
      

        Returns
        -------
        list
            list of dataframes of articles data
        """
        
        dff=[]
        for ticker in self.tickers:
            dff.append(self.get_articles(ticker))
        return dff
    
    