# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:07:29 2020

@author: Lenovo
"""

# Twitter Sentiment Analyzer

import tweepy
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.image as mpimg
import IPython

from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Navigation")
category = ["Twitter Data Analysis","Twitter Account Details","Source Code"]
choice = st.sidebar.radio("Navigation", category) 

st.sidebar.title("Created By:")
st.sidebar.subheader("Ashutosh Shinde")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/ashutoshashinde/)")
st.sidebar.subheader("[GitHub Repository](https://github.com/AshutoshAShinde/Twitter-Data-Analysis-App)")  

# Twitter Credentials

consumer_key = 'jDvT8Ydto8ABvgx0JsNqcOZsS'
consumer_secret = 'OTZkuey2d0YNxHIWZDp5O6vZ7A1bhod7KZzf1eOXV1Qxi8OwxG'

access_token = '1249542581660323845-JUclgVDUla2o1S0evqsi9C0sRsRWxF'
access_token_secret = 'FKMJkElmc0tFuGYYDxKCjKa6g6l2OFPL31sHfPZRiOn8v'

authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret) 
authenticate.set_access_token(access_token, access_token_secret) 

api = tweepy.API(authenticate, wait_on_rate_limit = True)
st.title('Twitter Data Analysis App')
images = mpimg.imread('trt.png',0)
st.image(images, height = 500, width = 500)
st.subheader("This Data App performs Analysis on any Twitter Account")

screen_name = st.text_area("Enter the exact twitter handle of the Personality (without @)")
st.write("Note: If you come across a ValueError, make sure to type the exact twitter handle (without @) and hit Ctrl + Enter")
st.graphviz_chart("""
        digraph{
        Tweet -> Sentiment
        Sentiment -> Positive
        Sentiment -> Neutral 
        Sentiment -> Negative
        Positive -> MostPositiveTweet
        Positive -> WordCloud
        Positive -> WordFrequency
        Negative -> MostNegativeTweet
        Negative -> WordCloud
        Negative -> WordFrequency
        Neutral -> WordCloud
        Neutral -> WordFrequency
        }
        """)
alltweets = []  
    
new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
alltweets.extend(new_tweets)
    
oldest = alltweets[-1].id - 1
    
while len(new_tweets) > 0:
        
    new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
            
for j in alltweets:
    
    totaltweets1 = j.user.statuses_count
    following1 = j.user.friends_count
    Description1 = j.user.description
    followers1 = j.user.followers_count
    Prof_image_url = j.user.profile_image_url

#st.write(Prof_image_url)
totaltweets = totaltweets1
following = following1
Description = Description1
followers = followers1


#image = plt.imread(Prof_image_url)
IPython.display.Image(Prof_image_url, width = 250)

twts = []
hshtgs = []
retweets = []

for tweet in alltweets:
    
    twts.append(tweet.text)
    retweets.append(tweet.retweet_count)
    hst = tweet.entities['hashtags']
    if len(hst) != 0:
        
        hh_list = []
        for i in range(len(hst)):
            dct = hst[i]
            hh_list.append(str(dct.get('text')))
        hshtgs.append(hh_list)
        
    else:
        hshtgs.append([])

pd.set_option('display.max_colwidth', None)
#pd.set_option('display.html.use_mathjax',False)
       

dict = {'Tweets': twts, 'Retweets':retweets}  
dfs = pd.DataFrame(dict)

dfs["word_count"] = dfs["Tweets"].apply(lambda tweet: len(tweet.split()))

senti_analyzer = SentimentIntensityAnalyzer()

compound_score = []

for sen in dfs['Tweets']:
    
    compound_score.append(senti_analyzer.polarity_scores(sen)['compound'])
    
dfs['Compound Score'] = compound_score

Sentiment = []

for i in compound_score:
    
    if i >= 0.05:
        
        Sentiment.append('Positive')
        
    elif i > -0.05 and i < 0.05:
        
        Sentiment.append('Neutral')
        
    else:
        
        Sentiment.append('Negative')
        
dfs['Sentiment'] = Sentiment

# Sentiment Distribution

positive_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])
negative_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])
neutral_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])

for i in range(len(dfs['Sentiment'])):
    
    if dfs.iloc[i]['Sentiment'] == 'Positive':
        positive_tweets = positive_tweets.append((dfs.iloc[i]))
        
    elif dfs.iloc[i]['Sentiment'] == 'Negative':
        negative_tweets = negative_tweets.append((dfs.iloc[i]))
        
    else:
        
        neutral_tweets = neutral_tweets.append((dfs.iloc[i]))

pos_count = sum(dfs['Sentiment']=='Positive')
neg_count = sum(dfs['Sentiment']=='Negative')
neu_count = sum(dfs['Sentiment']=='Neutral')

# Pie chart
labels = ['Positive Tweets', 'Negative Tweets', 'Neutral Tweets']
sizes = [pos_count, neg_count, neu_count]
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05)

# Most Positive Tweet

pos_max = dfs.loc[dfs['Compound Score']==max(dfs['Compound Score'])]

# Most Negative Tweet

neg_max = dfs.loc[dfs['Compound Score']==min(dfs['Compound Score'])]

def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordCloud)
    plt.axis("off")
    #plt.tight_layout(pad=0)
    #plt.show()

def wordcloud(data):
    
    words_corpus = ''
    words_list = []

    
    for rev in data["Tweets"]:
        
        text = str(rev).lower()
        text = text.replace('rt', ' ') 
        text = re.sub(r"http\S+", "", text)        
        text = re.sub(r'[^\w\s]','',text)
        text = ''.join([i for i in text if not i.isdigit()])
        
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        
        # Remove aplha numeric characters
        
        for words in tokens:
            
            words_corpus = words_corpus + words + " "
            words_list.append(words)
            
    return words_corpus, words_list   

positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_tweets)[0])
neutral_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(neutral_tweets)[0])
negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_tweets)[0])
total_wordcloud = WordCloud( width=900, height=500).generate(wordcloud(dfs)[0])
            
at = nltk.FreqDist(wordcloud(dfs)[1])
dt = pd.DataFrame({'Wordcount': list(at.keys()),
                  'Count': list(at.values())})
# selecting top 10 most frequent hashtags     
dt = dt.nlargest(columns="Count", n = 10)

# Most Frequent Words - Positive Tweets

ap = nltk.FreqDist(wordcloud(positive_tweets)[1])
dp = pd.DataFrame({'Wordcount': list(ap.keys()),
                  'Count': list(ap.values())})
# selecting top 10 most frequent hashtags     
dp = dp.nlargest(columns="Count", n = 10) 

# Most Frequent Words - Negative Tweets

an = nltk.FreqDist(wordcloud(negative_tweets)[1])
dn = pd.DataFrame({'Wordcount': list(an.keys()),
                  'Count': list(an.values())})
# selecting top 10 most frequent hashtags     
dn = dn.nlargest(columns="Count", n = 10) 

# Most Frequent Words - Neutral Tweets

au = nltk.FreqDist(wordcloud(neutral_tweets)[1])
du = pd.DataFrame({'Wordcount': list(au.keys()),
                  'Count': list(au.values())})
# selecting top 10 most frequent hashtags     
du = du.nlargest(columns="Count", n = 10)



if choice == "Twitter Account Details":
        
    st.subheader("The functions performed by the Twitter Sentiment Analysis Data App are :")    
    Functions = ["Display the Profile Image of this Twitter Account","Display the Most Recent Tweets of this Twitter Account","Description of this Twitter Account" ,"Number of Followers this Twitter Account has","The total number of tweets sent by this Twitter Account", "The number of Twitter Accounts followed by this Twitter Account"]
    Analyzer_choice = st.radio("Select the Function",Functions)	
		
    if Analyzer_choice == "Display the Profile Image of this Twitter Account":
            
        IPython.display.Image(Prof_image_url, width = 250)

    elif Analyzer_choice == "Display the Most Recent Tweets of this Twitter Account":
                
        st.write("The Most Recent Tweets are:") 
        st.table(dfs.head(7))
                
    elif Analyzer_choice == "Description of this Twitter Account":
        
        st.write("The desrciption of this Twitter Account is :")
        st.subheader(Description)

    elif Analyzer_choice == "The total number of tweets sent by this Twitter Account":
        
        st.write('The total number of tweets by this Twitter Account are:')
        st.subheader(totaltweets)

                
    elif Analyzer_choice == "Number of Followers this Twitter Account has":
                
        st.write('The number of followers this Twitter Account has are: ')
        st.subheader(followers)
        
    else:
        
        st.write('The number of Twitter Accounts followed by this Twitter Account are: ') 
        st.subheader(following)
            
elif choice =="Twitter Data Analysis":
    
    st.subheader("The functions performed by the Web App are :")

    st.write("1. Displays the most recent tweets")
    st.write("2. Displays the Sentiment Distribution of the tweets")
    st.write("3. Generates a wordcloud for all the tweets")
    st.write("4. Displays the Histogram for the Word Count Distribution of all the tweets")
    st.write("5. Displays the most frequent words used in all the tweets")

    Senti_Analyzer_choices1 = st.selectbox("Analysis Choice", ["Display the most recent tweets","Display the Sentiment Distribution of the tweets","Generate a wordcloud for all the tweets","Display the Histogram for the Word Count Distribution of all the tweets","Display the most frequent words used in all the tweets"])

    if st.button("Analyze"):
	
        if Senti_Analyzer_choices1 == "Display the most recent tweets":
            
            st.write("The Most Recent Tweets are:") 
            st.table(dfs.head())
            
        elif Senti_Analyzer_choices1 =="Display the Sentiment Distribution of the tweets":
                        
            st.write(plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode))
            st.pyplot(use_container_width=True)
                       
        elif Senti_Analyzer_choices1 =="Generate a wordcloud for all the tweets":
            
            st.write(plot_Cloud(total_wordcloud))
            st.pyplot(use_container_width=True)
            
        elif Senti_Analyzer_choices1 =="Display the Histogram for the Word Count Distribution of all the tweets":
            
            st.write(sns.distplot(dfs['word_count']))
            st.pyplot(use_container_width=True) 
            
        else:
            
            st.write('Most Frequent Words in all of the tweets')
            st.write(sns.barplot(data=dt, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)   
     
    st.write("6. Displays the most recent positive tweets")
    st.write("7. Generates a wordcloud for all the positive tweets")
    st.write("8. Displays the most positive tweet")
    st.write("9. Displays the most frequent words used in the positive tweets")
            
    Senti_Analyzer_choices2 = st.selectbox("Analysis Choice", ["Display the most recent positive tweets","Generate a wordcloud for all the positive tweets","Display the most positive tweet","Display the most frequent words used in the positive tweets"]) 
    
    if st.button("Analyze "):
                                       
        if Senti_Analyzer_choices2 =="Display the most recent positive tweets":
            
            st.write("The Most Recent Positive Tweets are:")
            st.write(positive_tweets.head())
       
        elif Senti_Analyzer_choices2 =="Generate a wordcloud for all the positive tweets":
                       
            st.write(plot_Cloud(positive_wordcloud))
            st.pyplot(use_container_width=True)
        
        elif Senti_Analyzer_choices2 =="Display the most positive tweet":
            
            st.write("Most positive tweet")     
            st.table(pos_max)
                
        else:
            
            st.write('Most Frequent Words in the positive tweets')
            st.write(sns.barplot(data=dp, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)
                   
    st.write("10. Displays the most recent negative tweets")
    st.write("11. Generates a wordcloud for all the negative tweets")
    st.write("12. Displays the most negative tweet")
    st.write("13. Displays the most frequent words used in the negative tweets") 
    
    Sent_Analyzer_choices3 = st.selectbox("Analysis Choices", ["Display the most recent negative tweets","Generate a wordcloud for all the negative tweets","Display the most negative tweet","Display the most frequent words used in the negative tweets"])

       
    if st.button("Analyze  "):
      
        if Sent_Analyzer_choices3 =="Display the most recent negative tweets":
            
            st.write("Most Recent Positive Tweets are:")
            st.table(negative_tweets.head())
        
        elif Sent_Analyzer_choices3 =="Generate a wordcloud for all the negative tweets":
            
            st.write("Wordcloud for all the Negative Tweets")
            st.write(plot_Cloud(negative_wordcloud))
            st.pyplot(use_container_width=True)
            
        elif Sent_Analyzer_choices3 =="Display the most negative tweet":
            
            st.write("Most negative tweet")
            st.table(neg_max)
        
        else:
            
            st.write('Most Frequent Words used in the negative tweets')
            st.write(sns.barplot(data=dn, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)
            
    st.write("14. Displays the most recent neutral tweets")   
    st.write("15. Generates a wordcloud for all the neutral tweets")
    st.write("16. Displays the most frequent words used in the neutral tweets")
        
    Senti_Analyzer_choices4 = st.selectbox("Analysis Choice", ["Display the most recent neutral tweets","Generate a wordcloud for all the neutral tweets","Display the most frequent words used in the neutral tweets"]) 
    
    if st.button("Analyze   "):                                                           
                                                               
        if Senti_Analyzer_choices4 == "Display the most recent neutral tweets":
            
            st.write("Most Recent Neutral Tweets are:")
            st.table(neutral_tweets.head())
            
        elif Senti_Analyzer_choices4 =="Generate a wordcloud for all the neutral tweets":
            
            st.write(plot_Cloud(neutral_wordcloud))
            st.pyplot(use_container_width=True)

        else:
            
            st.write('Most Frequent Words used in the neutral tweets')
            st.write(sns.barplot(data=du, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)                                                           
                                                               
                                                                                                                                                                                           
else:

        
    st.subheader("Source Code")
    
    code = """

import tweepy
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.image as mpimg

from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Twitter Credentials

consumer_key = 'jDvT8Ydto8ABvgx0JsNqcOZsS'
consumer_secret = 'OTZkuey2d0YNxHIWZDp5O6vZ7A1bhod7KZzf1eOXV1Qxi8OwxG'

access_token = '1249542581660323845-JUclgVDUla2o1S0evqsi9C0sRsRWxF'
access_token_secret = 'FKMJkElmc0tFuGYYDxKCjKa6g6l2OFPL31sHfPZRiOn8v'

authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret) 
authenticate.set_access_token(access_token, access_token_secret) 

api = tweepy.API(authenticate, wait_on_rate_limit = True)

st.title('Twitter Sentiment Analysis Data App')
images = mpimg.imread('trt.png')
st.image(images, height = 500, width = 500)
st.subheader("This Data App performs the Sentiment Analysis for any twitter account")

screen_name = st.text_area("Enter the exact twitter handle of the Personality (without @)")
st.write("Note: If you come across a ValueError below, make sure to type the username and hit Ctrl + Enter")
alltweets = []  
    
new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
alltweets.extend(new_tweets)
    
oldest = alltweets[-1].id - 1
    
while len(new_tweets) > 0:
        
    new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
            
for j in alltweets:
    
    totaltweets1 = j.user.statuses_count
    following1 = j.user.friends_count
    Description1 = j.user.description
    followers1 = j.user.followers_count
    Prof_image_url = j.user.profile_image_url


totaltweets = totaltweets1
following = following1
Description = Description1
followers = followers1

image = mpimg.imread(Prof_image_url)

twts = []
hshtgs = []
retweets = []

for tweet in alltweets:
    
    twts.append(tweet.text)
    retweets.append(tweet.retweet_count)
    hst = tweet.entities['hashtags']
    if len(hst) != 0:
        
        hh_list = []
        for i in range(len(hst)):
            dct = hst[i]
            hh_list.append(str(dct.get('text')))
        hshtgs.append(hh_list)
        
    else:
        hshtgs.append([])

pd.set_option('display.max_colwidth', None)
#pd.set_option('display.html.use_mathjax',False)
       

dict = {'Tweets': twts, 'Retweets':retweets}  
dfs = pd.DataFrame(dict)

dfs["word_count"] = dfs["Tweets"].apply(lambda tweet: len(tweet.split()))

senti_analyzer = SentimentIntensityAnalyzer()

compound_score = []

for sen in dfs['Tweets']:
    
    compound_score.append(senti_analyzer.polarity_scores(sen)['compound'])
    
dfs['Compound Score'] = compound_score

Sentiment = []

for i in compound_score:
    
    if i >= 0.05:
        
        Sentiment.append('Positive')
        
    elif i > -0.05 and i < 0.05:
        
        Sentiment.append('Neutral')
        
    else:
        
        Sentiment.append('Negative')
        
dfs['Sentiment'] = Sentiment

# Sentiment Distribution

positive_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])
negative_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])
neutral_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])

for i in range(len(dfs['Sentiment'])):
    
    if dfs.iloc[i]['Sentiment'] == 'Positive':
        positive_tweets = positive_tweets.append((dfs.iloc[i]))
        
    elif dfs.iloc[i]['Sentiment'] == 'Negative':
        negative_tweets = negative_tweets.append((dfs.iloc[i]))
        
    else:
        
        neutral_tweets = neutral_tweets.append((dfs.iloc[i]))

pos_count = sum(dfs['Sentiment']=='Positive')
neg_count = sum(dfs['Sentiment']=='Negative')
neu_count = sum(dfs['Sentiment']=='Neutral')

# Pie chart
labels = ['Positive Tweets', 'Negative Tweets', 'Neutral Tweets']
sizes = [pos_count, neg_count, neu_count]
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05)

# Most Positive Tweet

pos_max = dfs.loc[dfs['Compound Score']==max(dfs['Compound Score'])]

# Most Negative Tweet

neg_max = dfs.loc[dfs['Compound Score']==min(dfs['Compound Score'])]

def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordCloud)
    plt.axis("off")
    #plt.tight_layout(pad=0)
    #plt.show()

def wordcloud(data):
    
    words_corpus = ''
    words_list = []

    
    for rev in data["Tweets"]:
        
        text = str(rev).lower()
        text = text.replace('rt', ' ') 
        text = re.sub(r"http\S+", "", text)        
        text = re.sub(r'[^\w\s]','',text)
        text = ''.join([i for i in text if not i.isdigit()])
        
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        
        # Remove aplha numeric characters
        
        for words in tokens:
            
            words_corpus = words_corpus + words + " "
            words_list.append(words)
            
    return words_corpus, words_list   

positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_tweets)[0])
neutral_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(neutral_tweets)[0])
negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_tweets)[0])
total_wordcloud = WordCloud( width=900, height=500).generate(wordcloud(dfs)[0])
            
at = nltk.FreqDist(wordcloud(dfs)[1])
dt = pd.DataFrame({'Wordcount': list(at.keys()),
                  'Count': list(at.values())})
# selecting top 10 most frequent hashtags     
dt = dt.nlargest(columns="Count", n = 10)

# Most Frequent Words - Positive Tweets

ap = nltk.FreqDist(wordcloud(positive_tweets)[1])
dp = pd.DataFrame({'Wordcount': list(ap.keys()),
                  'Count': list(ap.values())})
# selecting top 10 most frequent hashtags     
dp = dp.nlargest(columns="Count", n = 10) 

# Most Frequent Words - Negative Tweets

an = nltk.FreqDist(wordcloud(negative_tweets)[1])
dn = pd.DataFrame({'Wordcount': list(an.keys()),
                  'Count': list(an.values())})
# selecting top 10 most frequent hashtags     
dn = dn.nlargest(columns="Count", n = 10) 

# Most Frequent Words - Neutral Tweets

au = nltk.FreqDist(wordcloud(neutral_tweets)[1])
du = pd.DataFrame({'Wordcount': list(au.keys()),
                  'Count': list(au.values())})
# selecting top 10 most frequent hashtags     
du = du.nlargest(columns="Count", n = 10)

category = ["Twitter Data Analysis","Twitter Account Details","Source Code"]
choice = st.sidebar.radio("Navigation", category)

if choice == "Twitter Account Details":
        
    st.subheader("The functions performed by the Twitter Sentiment Analysis Data App are :")    
    Functions = ["Display the Profile Image of this Twitter Account","Display the Most Recent Tweets of this Twitter Account","Description of this Twitter Account" ,"Number of Followers this Twitter Account has","The total number of tweets sent by this Twitter Account", "The number of Twitter Accounts followed by this Twitter Account"]
    Analyzer_choice = st.radio("Select the Function",Functions)	
		
    if Analyzer_choice == "Display the Profile Image of this Twitter Account":
            
        st.image(image, width = 100)

    elif Analyzer_choice == "Display the Most Recent Tweets of this Twitter Account":
                
        st.write("The Most Recent Tweets are:") 
        st.table(dfs.head(7))
                
    elif Analyzer_choice == "Description of this Twitter Account":
        
        st.write("The desrciption of this Twitter Account is :")
        st.subheader(Description)

    elif Analyzer_choice == "The total number of tweets sent by this Twitter Account":
        
        st.write('The total number of tweets by this Twitter Account are:')
        st.subheader(totaltweets)

                
    elif Analyzer_choice == "Number of Followers this Twitter Account has":
                
        st.write('The number of followers this Twitter Account has are: ')
        st.subheader(followers)
        
    else:
        
        st.write('The number of Twitter Accounts followed by this Twitter Account are: ') 
        st.subheader(following)
            
elif choice =="Twitter Data Analysis":
    
    st.subheader("The functions performed by the Web App are :")

    st.write("1. Displays the most recent tweets")
    st.write("2. Displays the Sentiment Distribution of the tweets")
    st.write("3. Generates a wordcloud for all the tweets")
    st.write("4. Displays the Histogram for the Word Count Distribution of all the tweets")
    st.write("5. Displays the most frequent words used in all the tweets")

    Senti_Analyzer_choices1 = st.selectbox("Analysis Choice", ["Display the most recent tweets","Display the Sentiment Distribution of the tweets","Generate a wordcloud for all the tweets","Display the Histogram for the Word Count Distribution of all the tweets","Display the most frequent words used in all the tweets"])

    if st.button("Analyze"):
	
        if Senti_Analyzer_choices1 == "Display the most recent tweets":
            
            st.write("The Most Recent Tweets are:") 
            st.table(dfs.head())
            
        elif Senti_Analyzer_choices1 =="Display the Sentiment Distribution of the tweets":
                        
            st.write(plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode))
            st.pyplot(use_container_width=True)
                       
        elif Senti_Analyzer_choices1 =="Generate a wordcloud for all the tweets":
            
            st.write(plot_Cloud(total_wordcloud))
            st.pyplot(use_container_width=True)
            
        elif Senti_Analyzer_choices1 =="Display the Histogram for the Word Count Distribution of all the tweets":
            
            st.write(sns.distplot(dfs['word_count']))
            st.pyplot(use_container_width=True) 
            
        else:
            
            st.write('Most Frequent Words in all of the tweets')
            st.write(sns.barplot(data=dt, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)   
     
    st.write("6. Displays the most recent positive tweets")
    st.write("7. Generates a wordcloud for all the positive tweets")
    st.write("8. Displays the most positive tweet")
    st.write("9. Displays the most frequent words used in the positive tweets")
            
    Senti_Analyzer_choices2 = st.selectbox("Analysis Choice", ["Display the most recent positive tweets","Generate a wordcloud for all the positive tweets","Display the most positive tweet","Display the most frequent words used in the positive tweets"]) 
    
    if st.button("Analyze "):
                                       
        if Senti_Analyzer_choices2 =="Display the most recent positive tweets":
            
            st.write("The Most Recent Positive Tweets are:")
            st.write(positive_tweets.head())
       
        elif Senti_Analyzer_choices2 =="Generate a wordcloud for all the positive tweets":
                       
            st.write(plot_Cloud(positive_wordcloud))
            st.pyplot(use_container_width=True)
        
        elif Senti_Analyzer_choices2 =="Display the most positive tweet":
            
            st.write("Most positive tweet")     
            st.table(pos_max)
                
        else:
            
            st.write('Most Frequent Words in the positive tweets')
            st.write(sns.barplot(data=dp, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)
                   
    st.write("10. Displays the most recent negative tweets")
    st.write("11. Generates a wordcloud for all the negative tweets")
    st.write("12. Displays the most negative tweet")
    st.write("13. Displays the most frequent words used in the negative tweets") 
    
    Sent_Analyzer_choices3 = st.selectbox("Analysis Choices", ["Display the most recent negative tweets","Generate a wordcloud for all the negative tweets","Display the most negative tweet","Display the most frequent words used in the negative tweets"])

       
    if st.button("Analyze  "):
      
        if Sent_Analyzer_choices3 =="Display the most recent negative tweets":
            
            st.write("Most Recent Positive Tweets are:")
            st.table(negative_tweets.head())
        
        elif Sent_Analyzer_choices3 =="Generate a wordcloud for all the negative tweets":
            
            st.write("Wordcloud for all the Negative Tweets")
            st.write(plot_Cloud(negative_wordcloud))
            st.pyplot(use_container_width=True)
            
        elif Sent_Analyzer_choices3 =="Display the most negative tweet":
            
            st.write("Most negative tweet")
            st.table(neg_max)
        
        else:
            
            st.write('Most Frequent Words used in the negative tweets')
            st.write(sns.barplot(data=dn, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)
            
    st.write("14. Displays the most recent neutral tweets")   
    st.write("15. Generates a wordcloud for all the neutral tweets")
    st.write("16. Displays the most frequent words used in the neutral tweets")
        
    Senti_Analyzer_choices4 = st.selectbox("Analysis Choice", ["Display the most recent neutral tweets","Generate a wordcloud for all the neutral tweets","Display the most frequent words used in the neutral tweets"]) 
    
    if st.button("Analyze   "):                                                           
                                                               
        if Senti_Analyzer_choices4 == "Display the most recent neutral tweets":
            
            st.write("Most Recent Neutral Tweets are:")
            st.table(neutral_tweets.head())
            
        elif Senti_Analyzer_choices4 =="Generate a wordcloud for all the neutral tweets":
            
            st.write(plot_Cloud(neutral_wordcloud))
            st.pyplot(use_container_width=True)

        else:
            
            st.write('Most Frequent Words used in the neutral tweets')
            st.write(sns.barplot(data=du, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)       

"""
    st.code(code, language='python')
    

     
    




    

        



