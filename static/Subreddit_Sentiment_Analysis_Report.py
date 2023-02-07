#!/usr/bin/env python
# coding: utf-8

# # Reddit authorization using Praw API
# 
# Getting comments from posts under a specific subreddit, then creates a sentimental analysis report of those comments
def doSentimentReport(user_input):
  return_list = []
  # In[1]:
  import sys
  import praw
  import nltk
  import pandas as pd
  from nltk.tokenize import word_tokenize
  import re #regex for cleaning text
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set(style='darkgrid', context='talk')
  from dotenv import load_dotenv
  import os

  print("Goes to file correctly", file=sys.stderr)

  # In[2]:

  load_dotenv()

  CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
  CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
  USER_AGENT = os.getenv('REDDIT_USER_AGENT')


  try:
      reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
      print("Authentication Successful")
  except:
      # reset all credentials
      CLIENT_ID = ''
      CLIENT_SECRET = ''
      USER_AGENT = ''
      print("Error: Authentication Failed")
      return (CLIENT_ID, CLIENT_SECRET, -1)

  # In[20]:

  input_subreddit = user_input

  # analyze movies subreddit
  subreddit = reddit.subreddit(input_subreddit).hot(limit=50)


  # In[21]:

  # create pandas dataframe
  posts = []

  # Test if the subreddit actually exists
  try: 
    for post in subreddit:
      posts.append([post.title, post.score, post.id, post.selftext, f"reddit.com{post.permalink}", post.num_comments])
  except: print("The subreddit does not exist or is not accessible."); return None # Exits the file immediately if subreddit doesn't exist

  posts = pd.DataFrame(posts,columns=['Title', 'Score', 'ID', 'Text', 'Post URL', 'Total Comments'])

  posts


  # In[22]:


  from praw.models import MoreComments
  import nltk.corpus
  nltk.download('stopwords')
  from nltk.corpus import stopwords

  # Get comments for each post
  post_comments = []
  all_comments = []
  ID = []

  print("This can take a while...")

  for post_id,post_url in zip(posts['ID'], posts['Post URL']):
    print(post_url)

    comment_text = ""

    # creating a submission object
    submission = reddit.submission(id=post_id)

    # take all comments of each submission
    submission.comments.replace_more(limit=1)
    for submission_comment in submission.comments:

      comment_text = f"{comment_text} {submission_comment.body}"

      all_comments.append(submission_comment.body)
    
    post_comments.append(comment_text)

  # creating a new dataframe of the comments
  df_comments = pd.DataFrame(post_comments, columns=['comment'])

  data_tuple = list(all_comments)
  df_all_comments = pd.DataFrame(data_tuple, columns=['comment'])

  df_comments


  # # Data preprocessing

  # In[23]:


  # checking that all of the comments are shown correctly
  df_all_comments


  # In[24]:


  df_comments['comment'][3]


  # In[25]:


  from nltk.corpus import stopwords
  # cleaning comment so that it will be appropriate to analyze
  stop = stopwords.words('english')

  def clean_comments(comment):
    c = comment
    c = re.sub(r'\n+', '', c)
    c = re.sub(r'\'', '', c)
    c = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", comment)
    c = c.strip().lower()
    c = " ".join([word for word in c.split() if word not in (stop)])
    return c

  comment_cleaned = []

  for comment in df_all_comments['comment']:
    cleaned = clean_comments(comment)

    comment_cleaned.append(cleaned)

  # make new dataframe for the cleaned comments 
  df_cleaned = pd.DataFrame(comment_cleaned, columns=['comment'])
  # drop rows which is just an empty string and then reset index
  df_cleaned = df_cleaned[df_cleaned.comment != ''].reset_index()
  df_cleaned


  # In[26]:


  df_cleaned['comment'][3]


  # In[27]:


  # Make dataframe into list for it to be tokenizable
  df_cleanedList = list(df_cleaned['comment'])
  df_cleanedList = ' '.join(df_cleanedList)
  df_cleanedList


  # # Tokenization, Lemmatization and Frequency Distribution

  # In[28]:


  from nltk.probability import FreqDist
  nltk.download('punkt')
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer
  nltk.download('wordnet')
  nltk.download('omw-1.4')

  # Tokenize words
  tokens = word_tokenize(df_cleanedList)

  lemmatized_tokens = []
  # Lemmatize words
  lemmatizer = WordNetLemmatizer()
  for token in tokens:
    lemmatized_tokens.append(lemmatizer.lemmatize(token))

  # Show frequency of words
  fd = FreqDist(lemmatized_tokens)

  fd


  # In[29]:


  cleanWords = nltk.Text(lemmatized_tokens)
  cleanWords.vocab().tabulate(10)


  df_fdist = pd.DataFrame.from_dict(fd, orient='index')
  df_fdist.columns = ['Frequency']
  df_fdist = df_fdist.nlargest(10, 'Frequency')
  df_fdist.index

  return_list.append(df_fdist)

  # # N-grams
  # Searching for most common trigrams used

  # In[30]:


  from nltk.collocations import TrigramCollocationFinder

  words = [w for w in cleanWords if w.isalpha()]
  finder = TrigramCollocationFinder.from_words(words)

  # Find the 20 most common trigrams
  finder.ngram_fd.most_common(20)



  # # Sentiment Analysis

  # In[31]:


  nltk.download('vader_lexicon')


  # In[32]:


  df_cleaned


  # In[33]:


  from nltk.sentiment import SentimentIntensityAnalyzer

  sia = SentimentIntensityAnalyzer()

  positive = []
  negative = []
  neutral = []
  compound = []

  for comment in df_cleaned['comment']:

    positive.append(sia.polarity_scores(comment)['pos'])
    negative.append(sia.polarity_scores(comment)['neg'])
    neutral.append(sia.polarity_scores(comment)['neu'])
    compound.append(sia.polarity_scores(comment)['compound'])

  df_cleaned['positive'] = positive
  df_cleaned['negative'] = negative
  df_cleaned['neutral'] = neutral
  df_cleaned['compound'] = compound


  # In[34]:


  df_cleaned


  # In[36]:


  # Make labels for the different subreddits using compound, if positive = 1, if negative = -1, if neutral = 0
  df_cleaned['label'] = 0
  df_cleaned.loc[df_cleaned['compound'] > 0.2, 'label'] = 1
  df_cleaned.loc[df_cleaned['compound'] < -0.2, 'label'] = -1


  pos_total = df_cleaned.loc[df_cleaned['label'] == 1]
  neg_total = df_cleaned.loc[df_cleaned['label'] == -1]
  neu_total = df_cleaned.loc[df_cleaned['label'] == 0]
  #total = pos_total + neg_total + neu_total
  total = len(pos_total) + len(neg_total) + len(neu_total)

  pos_ratio = len(pos_total)/total
  neg_ratio = len(neg_total)/total
  neu_ratio = len(neu_total)/total

  # take only one decimal place
  print("Percentage of comments which are considered positive: {:.1f} %".format(100.0*pos_ratio))

  print("Percentage of comments which are considered negative: {:.1f} %".format(100.0*neg_ratio))

  print("Percentage of comments which are considered neutral: {:.1f} %".format(100.0*neu_ratio))


  # In[37]:
  df_cleaned = df_cleaned.drop(columns=['index'])

  return_list.append(df_cleaned)


  # returns the top 10 frequency distribution of words in pandas dataframe,
  # the top 20 trigrams
  # dataframe with sentiment
  return return_list



  # In[38]:


  import seaborn as sns
  sns.set(style='darkgrid', context='talk', palette='Dark2')


  # In[39]:


  # visualize the percentages
  fig, ax = plt.subplots(figsize=(8, 8))

  counts = df_cleaned.label.value_counts(normalize=True) * 100.0

  sns.barplot(x=counts.index, y=counts, ax=ax)

  ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
  ax.set_ylabel("Percentage")

  plt.show()

  #define data
  fig, ax = plt.subplots(figsize=(8, 8))

  data = counts
  labels = []

  # rename labels from numeric to its correct string
  for label in counts.index:
    if label == 0:
      label = 'Neutral'
    if label == 1:
      label = 'Positive'
    if label == -1:
      label = 'Negative'
    labels.append(label)

  #define Seaborn color palette to use
  colors = sns.color_palette('pastel')[0:5]

  #create pie chart
  plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
  plt.show()
