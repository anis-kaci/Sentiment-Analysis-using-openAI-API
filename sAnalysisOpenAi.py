'''
    Sentiment analysis using openAi API
'''
#imports 
import pandas as pd
import openai 
import os 

import re 
#for metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#set up the openai API client
openai.api_key = "YOUR OPENAI KEY"

#example of function proving s analysis using openAi api 
def get_sentiment(text):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "user", "content": f"Sentiment analysis of the following text: {text}"}
    ]
    )
    sent = completion.choices[0].message
    return sent

#print(get_sentimentdos("I love you"))



#check response string to number 
def find_sentiment(text):
    # Define a regular expression pattern to match any of the specified words
    pattern = r"\b(positive|negative|neutral)\b"

    # Search for the first match
    match = re.search(pattern, text, re.IGNORECASE)  # Case-insensitive search

    if match:
        return match.group()
    else:
        return None  # Return None if no match is found

def responseToNumber(oAiObj):
    text = oAiObj['content']
    sent = find_sentiment(text)

    if(sent == "positive"):
        return 1 
    elif(sent == "negative"):
        return 0 
    elif(sent == "neutral"):
        return 4 
    
    #just in case 
    return 9




df = pd.read_csv('Datasets/training.1600000.processed.noemoticon.csv',
                 delimiter=',', encoding='ISO-8859-1')
df.columns = ['Sentiment','id','date','query','user','text']
df = df[['Sentiment','text']]


#reduce df for testing 
#df = df.sample(10000)


true_res = []
predicted = []

for index, row in df.iterrows():
    text = row['text']
    true_sentiment = row['Sentiment'] #actual sentiment from the dataset 
    predicted_sentiment = get_sentiment(text)

    true_res.append(true_sentiment)
    predicted.append(responseToNumber(predicted_sentiment))



#print(true_res)
#print(predicted)


#metrics
accuracy = accuracy_score(true_res, predicted)
precision = precision_score(true_res, predicted, average='weighted')
recall = recall_score(true_res, predicted, average = 'weighted')
f1 = f1_score(true_res, predicted, average='weighted')



print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)


