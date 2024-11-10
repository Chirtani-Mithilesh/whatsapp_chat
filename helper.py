from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
extract = URLExtract()
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Path to the Segoe UI Emoji font on Windows
font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"  # Ensure the path is correct for your system
prop = font_manager.FontProperties(fname=font_path)
# Set the font globally in matplotlib
plt.rcParams["font.family"] = prop.get_name()

# Function to fetch general statistics about the user's chat
def fetch_stats(selected_user, df):
    user_df = df[df['user'] == selected_user] if selected_user != "Overall" else df
    
    total_messages = user_df.shape[0]
    total_words = user_df['message'].apply(lambda x: len(x.split())).sum()
    total_emoji_count = user_df['emoji_count'].sum()
    avg_message_length = user_df['message_length'].mean()
    avg_sentiment = user_df['sentiment'].mean()

    stats = {
        "Total Messages": total_messages,
        "Total Words": total_words,
        "Total Emojis": total_emoji_count,
        "Average Message Length": avg_message_length,
        "Average Sentiment": avg_sentiment
    }

    return stats

# Function to prepare the dataset for training and testing
def prepare_data(df):
    features = df[['message_length', 'sentiment', 'adjusted_sentiment', 'negativity_ratio', 'self_referencing', 
                   'emoji_count', 'happy_emoji_count', 'sad_emoji_count']]
    
    def calculate_mental_health(row):
        sentiment_weight = -0.5
        negativity_weight = 0.3
        self_ref_weight = 0.2
        happy_emoji_weight = -0.3
        sad_emoji_weight = 0.4

        score = (row['sentiment'] * sentiment_weight +
                 row['adjusted_sentiment'] * sentiment_weight +
                 row['negativity_ratio'] * negativity_weight +
                 row['self_referencing'] * self_ref_weight +
                 row['happy_emoji_count'] * happy_emoji_weight +
                 row['sad_emoji_count'] * sad_emoji_weight)

        return 1 if score > 0.5 else 0
    
    df['mental_health'] = df.apply(calculate_mental_health, axis=1)
    target = df['mental_health']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to train the Random Forest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model's performance
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    eval_metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Classification Report": classification_report(y_test, predictions, output_dict=True)
    }
    return eval_metrics

# Function to predict mental health state and explain with SHAP
def predict_and_explain(recent_df, model):
    # Select the features for prediction
    features = recent_df[['message_length', 'sentiment', 'adjusted_sentiment', 'negativity_ratio', 
                          'self_referencing', 'emoji_count', 'happy_emoji_count', 'sad_emoji_count']]
    
    # Make predictions
    predictions = model.predict(features)
    
    # Initialize SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    return predictions, shap_values


# Function to plot SHAP explanation
def plot_shap_explanation(shap_values):
    fig = plt.figure()
    max_features = shap_values[1].shape[1]  # Get the number of features from shap_values
    shap.summary_plot(shap_values[1], plot_type="bar", show=False, max_display=max_features)
    plt.tight_layout()
    return fig


# Function to generate user feedback based on predictions
def generate_user_feedback(recent_df, predictions):
    feedback = []
    for i, pred in enumerate(predictions):
        message = recent_df.iloc[i]['message']
        sentiment = recent_df.iloc[i]['sentiment']
        if pred == 1:
            feedback.append(f"Warning: Message '{message}' shows signs of distress. Sentiment: {sentiment}")
        elif sentiment>=0:
            feedback.append(f"Positive: Message '{message}' shows positive sentiment. Sentiment: {sentiment}")
    return feedback

# Function to find most active users in group chat
def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df

# Function to create word cloud
def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        return " ".join([word for word in message.lower().split() if word not in stop_words])

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

# Function to find most common words
def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = [word for message in temp['message'] for word in message.lower().split() if word not in stop_words]

    return pd.DataFrame(Counter(words).most_common(20))

# Function for emoji analysis
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = [c for message in df['message'] for c in message if c in emoji.EMOJI_DATA]
    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

# Function to generate monthly timeline
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline

# Function to generate daily timeline
def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()

# Function to map week activity
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

# Function to map month activity
def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

# Function to create activity heatmap
def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
