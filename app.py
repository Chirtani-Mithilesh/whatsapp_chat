import streamlit as st
import preprocessor
import helper
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib import font_manager

# Set up font for displaying emojis
matplotlib.use('Agg')
font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"  # Ensure the path is correct for your system
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = prop.get_name()

# Streamlit sidebar setup
st.sidebar.title("WhatsApp Chat Mental Health Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file")
if uploaded_file is not None:
    # Preprocess the chat data
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    
    # Prepare the data for training (train-test split)
    X_train, X_test, y_train, y_test = helper.prepare_data(df)
    
    # Train the Random Forest model
    model = helper.train_model(X_train, y_train)
    
    # Evaluate the model
    eval_metrics = helper.evaluate_model(model, X_test, y_test)
    st.write("Model Accuracy:", eval_metrics["Accuracy"])
    st.write("Classification Report:", eval_metrics["Classification Report"])

    # Fetch unique users for analysis
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Select user for analysis", user_list)
    message_count = st.sidebar.slider("Select number of recent messages", 10, 100, 20)

    if st.sidebar.button("Show Analysis"):
        st.title(f"Analysis for {selected_user}")
        
        # Display general statistics
        stats = helper.fetch_stats(selected_user, df)
        st.write("Top Statistics:")
        for stat, value in stats.items():
            st.write(f"{stat}: {value}")

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)
        st.pyplot(fig)

        # Busiest Users (Group Level)
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Word Cloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most Common Words")
        st.pyplot(fig)

        # Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        # Predict Mental Health State for Recent Messages        
        if selected_user != "Overall":
            # Fetch recent messages for the selected user
            recent_df = df[df['user'] == selected_user].tail(message_count)
            predictions, shap_values = helper.predict_and_explain(recent_df, model)

            st.title(f"Predicted Mental Health State for Last {message_count} Messages")
            st.write(predictions)

            # SHAP Explainability Visualization
            st.title("Explainability: SHAP Values")
            shap.initjs()
            st.write("SHAP values for the predictions:")
            st.pyplot(helper.plot_shap_explanation(shap_values))
            # Generate and Display User Feedback Based on Predictions
            st.title("User  Feedback")
            feedback_messages = helper.generate_user_feedback(recent_df, predictions)
            for feedback in feedback_messages:
                st.write(feedback)
