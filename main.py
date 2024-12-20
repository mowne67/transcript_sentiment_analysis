import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@st.cache_resource
def load_vader_analyzer():
    return SentimentIntensityAnalyzer()


vader_analyzer = load_vader_analyzer()

st.title("Call Transcript Sentiment Analyzer with VADER")
file = st.file_uploader("Upload a txt file", type = ['txt'])
if file is not None:
    file_content = file.read().decode('utf-8')
    st.subheader('File Content:')
    st.text(file_content)

    transcript = file_content
    call_duration = st.number_input("Enter the total duration of the call (in minutes):", min_value=0, step=1)

    if call_duration > 0:
        selected_time = st.slider(
            "Select the time range to analyze:",
            min_value=0,
            max_value=int(call_duration * 60),
            value=(0, int(call_duration * 60)),
            step=1,
            format="%d seconds",
        )

        start_time, end_time = selected_time
        words_per_second = 2  # Estimate of words per second
        start_idx = start_time * words_per_second
        end_idx = end_time * words_per_second

        subset_transcript = " ".join(transcript.split()[int(start_idx):int(end_idx)])

        st.subheader("Selected Transcript Section:")
        st.write(subset_transcript if subset_transcript else "No transcript for the selected time range.")

        if subset_transcript:
            sentiment_scores = vader_analyzer.polarity_scores(subset_transcript)

            compound_score = sentiment_scores["compound"]
            sentiment = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"

            st.subheader("Sentiment Analysis Result:")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Score:** {compound_score}")
