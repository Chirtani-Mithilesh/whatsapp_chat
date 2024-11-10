import pandas as pd
import re
from datetime import datetime
from textblob import TextBlob
import emoji

# Define an emoji sentiment dictionary with categories for positive, negative, and other emotions
emoji_sentiment = {
    '😀': 0.5, '😃': 0.5, '😄': 0.5, '😁': 0.4, '😆': 0.5, '😂': 0.4,
    '🤣': 0.4, '😊': 0.4, '😍': 0.6, '🥰': 0.6, '😘': 0.5, '😗': 0.4,
    '😙': 0.4, '😚': 0.4, '😜': 0.3, '😝': 0.3, '🤪': 0.3, '😎': 0.4,
    '😌': 0.3, '😋': 0.3, '🤩': 0.5, '😺': 0.5,  # Positive emojis

    '😢': -0.5, '😭': -0.6, '😞': -0.4, '😔': -0.4, '😟': -0.4, '😕': -0.3,
    '😫': -0.5, '😩': -0.5, '😣': -0.4, '😖': -0.4, '😥': -0.4, '😓': -0.3,
    '😰': -0.5, '😨': -0.4, '🙁': -0.3, '☹️': -0.4, '🥲': -0.3,  # Negative emojis

    '😡': -0.6, '😠': -0.5, '🤬': -0.7, '😤': -0.5,  # Angry emojis
    '😱': -0.4, '😨': -0.4, '😳': -0.3, '🥵': -0.4,  # Surprised/anxious emojis
}

def preprocess(data):
    # Define the pattern to match WhatsApp messages
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s[APM]{2}) - (.*?): (.*)'
    messages = re.findall(pattern, data)
    
    # Initialize lists to store extracted data
    dates, times, users, msgs = [], [], [], []

    # Process each matched message
    for date, time, user, message in messages:
        dates.append(date)
        times.append(time)
        users.append(user)
        msgs.append(message)
    
    # Create a DataFrame from extracted data
    df = pd.DataFrame({'date': dates, 'time': times, 'user': users, 'message': msgs})
    df = df[~df['message'].str.contains('<Media omitted>')]  # Filter out media messages

    # Convert 'date' and 'time' columns to datetime format
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%y %I:%M %p')
    df['only_date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.month_name()
    df['day'] = df['datetime'].dt.day
    df['day_name'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    # Message length (handle empty messages)
    df['message_length'] = df['message'].apply(lambda x: len(x) if isinstance(x, str) else 0)

    # Sentiment Analysis
    df['sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)
    
    # Adjust sentiment with emoji mapping
    def adjust_sentiment_with_emojis(text):
        sentiment_score = TextBlob(text).sentiment.polarity if text else 0
        emoji_adjustment = sum(emoji_sentiment.get(char, 0) for char in text)
        return sentiment_score + emoji_adjustment
    
    df['adjusted_sentiment'] = df['message'].apply(adjust_sentiment_with_emojis)

    # Count Negative Emotion Words
    negative_words = set([
        'angry', 'sad', 'depressed', 'lonely', 'hate', 'miserable', 'hurt', 'betrayed', 'anxious', 'terrible', 'bad', 'worse', 
        'stressed', 'hopeless', 'frustrated', 'unhappy', 'disappointed', 'negative', 'agitated', 'disgusted', 'frightened', 
        'rejected', 'isolated', 'unworthy', 'painful', 'guilty', 'scared', 'worthless', 'empty', 'awful', 'insecure', 
        'overwhelmed', 'paranoid', 'fearful', 'irritable', 'concerned', 'panic', 'annoyed', 'miserable', 'hopeless', 
        'nervous', 'frustration', 'vulnerable', 'distressed', 'unstable', 'hurtful', 'crying', 'grief', 'confused', 'rejected', 
        'dread', 'tired', 'helpless', 'terrified', 'regret', 'shattered', 'broken', 'despair', 'frantic', 'worried', 'uncertain', 
        'conflicted', 'exhausted', 'burdened', 'disturbed', 'bitter', 'lost', 'sick', 'frustrating', 'pain', 'unforgiven', 
        'scorned', 'worried', 'tearful', 'reckless', 'distress', 'anxiety', 'unsettled', 'doubt', 'mournful', 'disarray', 
        'neglect', 'rage', 'irritation', 'discomfort', 'shame', 'grief-stricken', 'defeated', 'defensive', 'tension', 'upset', 
        'apprehensive', 'skeptical', 'displeased', 'lament', 'withdrawn', 'overcome', 'despondent', 'sorrow', 'inconsolable', 
        'apprehension', 'desolate', 'embarrassed', 'gloomy', 'frustrate', 'anguish', 'resentment', 'crushed', 'depress', 
        'dejected', 'mourn', 'unhappy', 'melancholy'
    ])
    df['negativity_ratio'] = df['message'].apply(lambda x: sum(1 for word in x.lower().split() if word in negative_words) / len(x.split()) if x else 0)

    # Count Self-Referencing Words
    self_referencing_words = {'i', 'me', 'my', 'myself'}
    df['self_referencing'] = df['message'].apply(lambda x: sum(1 for word in x.lower().split() if word in self_referencing_words) if x else 0)

    # Emoji Analysis
    def count_emojis(text):
        return sum(1 for char in text if char in emoji.EMOJI_DATA)
    
    df['emoji_count'] = df['message'].apply(lambda x: count_emojis(x) if x else 0)
    df['happy_emoji_count'] = df['message'].apply(lambda x: sum(1 for char in x if char in '😀😃😄😁😆😂🤣😊😇🥰😍😘😗😙😚😜😝🤪😎😌😋🥳🎉🤗🤩😺') if x else 0)
    df['sad_emoji_count'] = df['message'].apply(lambda x: sum(1 for char in x if char in '😢😭😿😞😔😟😕😫😩🥺😣😖😥😓😰😨🙁☹️🥲') if x else 0)
    df['angry_emoji_count'] = df['message'].apply(lambda x: sum(1 for char in x if char in '😡😠🤬😤') if x else 0)
    df['surprise_emoji_count'] = df['message'].apply(lambda x: sum(1 for char in x if char in '😱😨😳🥵') if x else 0)
    # Define time periods
    df['period'] = df['hour'].apply(lambda hour: f"{hour}-00" if hour == 23 else (f"00-1" if hour == 0 else f"{hour}-{hour+1}"))
    return df
