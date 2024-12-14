import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image
import requests
import torch
import os
import tempfile
import base64
import plotly.express as px
import pandas as pd

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="✨ Sentiment & Emotion Analysis App ✨",
    page_icon="✨",
    layout="wide",  # Use the full width of the page
    initial_sidebar_state="expanded",
)

# ------------------------------
# Custom CSS for Enhanced Aesthetics
# ------------------------------
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" integrity="sha512-Fo3rlrZj/k7ujTnHg4CGR2DkKE6Y4jC+I1clgZK5QaJgFhC1Q1vy+M1X+lg6mgpUq0gK0hYqIbua4Zr4Xr3P0w==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* Global Fonts and Colors */
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            color: #333333;
        }
        
        /* Main Title */
        h1 {
            color: #2c3e50;
            text-align: center;
            font-size: 48px;
            margin-bottom: 10px;
            animation: fadeInDown 1s ease-in-out;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555555;
            margin-bottom: 40px;
            animation: fadeInUp 1s ease-in-out;
        }

        /* Button Styles */
        .stButton > button {
            color: white;
            background: linear-gradient(45deg, #2980b9, #3498db);
            border: none;
            padding: 12px 24px;
            font-size: 18px;
            border-radius: 25px;
            transition: background 0.3s ease, transform 0.2s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton > button:hover {
            background: linear-gradient(45deg, #3498db, #2980b9);
            transform: translateY(-2px);
        }

        /* Tab Styles */
        .css-1aumxhk {  /* Specific to Streamlit's tab styling */
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-in-out;
        }

        /* Active Tabs */
        .stTabs [data-baseweb="tab"] [aria-selected="true"] > div {
            background-color: #2980b9;
            color: white;
            border-radius: 5px 5px 0 0;
            font-weight: bold;
        }
        .stTabs [data-baseweb="tab"] [aria-selected="false"] > div {
            background-color: #ecf0f1;
            color: #2c3e50;
            border-radius: 5px 5px 0 0;
        }

        /* Tabs Container */
        .stTabs {
            margin-bottom: 20px;
        }

        /* Image Hover Effect */
        .hover-image {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-radius: 10px;
            max-width: 100%;
            height: auto;
        }
        .hover-image:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }

        /* Results Styling */
        .results {
            font-size: 18px;
            color: #333333;
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-top: 20px;
            animation: slideIn 0.5s ease-in-out;
        }

        /* Progress Bar */
        div.stProgress > div > div {
            background-color: #2980b9;
        }

        /* Footer */
        footer {
            text-align: center;
            font-size: 14px;
            color: gray;
            margin-top: 50px;
        }

        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideIn {
            from { transform: translateX(-50px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            h1 {
                font-size: 36px;
            }
            .subtitle {
                font-size: 16px;
            }
            .stButton > button {
                font-size: 16px;
                padding: 10px 20px;
            }
            .results {
                font-size: 16px;
            }
        }

        /* Tooltip Styling */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
            margin-left: 5px;
            color: #2980b9;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #2c3e50;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the text */
            left: 50%;
            margin-left: -100px; /* Center the tooltip */
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Sidebar for General Information and Navigation
# ------------------------------
with st.sidebar:
    # Uncomment and replace with your logo URL if needed
    # st.image("https://i.imgur.com/4AI6h0U.png", width=150)
    st.title("Sentiment & Emotion Analysis")
    st.markdown("### **Enhance Your Understanding**")
    st.markdown(
        """
        Welcome to the **Sentiment & Emotion Analysis App**! Use the tabs on the main interface to analyze text, images, audio, multimodal inputs, or Twitter data.
        
        **Features:**
        - 📝 Analyze the sentiment of your text.
        - 🖼️ Detect emotions in your images.
        - 🔊 Understand emotions & sentiment in your audio.
        - 🔗 Combine multiple inputs for comprehensive analysis.
        - 🐦 Perform sentiment analysis on Twitter data.
        
        **Get Started:**
        - Select a tab and provide the necessary inputs.
        - View results with interactive charts and detailed insights.
        """
    )
    

# ------------------------------
# App Title
# ------------------------------
st.markdown("<h1>✨ Sentiment & Emotion Analysis App ✨</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Analyze the <strong>sentiment</strong> of your text, the <strong>emotions</strong> in your images, and the <strong>emotions & sentiment</strong> in your audio.</p>",
    unsafe_allow_html=True,
)

# ------------------------------
# Creating Tabs
# ------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Text Sentiment Analysis",
    "🖼️ Image Emotion Analysis",
    "🔊 Audio Emotion & Sentiment Analysis",
    "🔗 Multimodal Analysis",
    "🐦 Twitter Sentiment Analysis"
])

# ------------------------------
# Utility Functions
# ------------------------------

@st.cache_resource
def load_text_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def analyze_sentiment(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities.tolist()

def send_request(endpoint, data=None, files=None, method='POST'):
    try:
        if method.upper() == 'POST':
            response = requests.post(endpoint, data=data, files=files)
        elif method.upper() == 'GET':
            response = requests.get(endpoint, params=data)
        else:
            st.error(f"❌ Unsupported HTTP method: {method}")
            return None
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to connect to the backend: {e}")
        return None

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"❌ Failed to encode file to base64: {e}")
        return ""

def get_emotion_emoji(emotion):
    emotion_emojis = {
        "happy": "😊",
        "sad": "😞",
        "angry": "😡",
        "fearful": "😨",
        "disgust": "🤢",
        "surprised": "😲",
        "neutral": "😐"
    }
    return emotion_emojis.get(emotion.lower(), "🤔")  # Default emoji

def download_results(data, filename):
    if isinstance(data, dict):
        # For JSON-like data
        csv = pd.json_normalize(data).to_csv(index=False)
    elif isinstance(data, list):
        # For list of dictionaries
        csv = pd.DataFrame(data).to_csv(index=False)
    else:
        # For other types, convert to string
        csv = pd.DataFrame({'Result': [str(data)]}).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Results</a>'
    return href

# ------------------------------
# Helper Function to Center Plotly Charts
# ------------------------------
def center_plotly_chart(fig, width=600, height=None):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig, use_container_width=True, width=width, height=height)

# ------------------------------
# TEXT SENTIMENT ANALYSIS
# ------------------------------
with tab1:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("📝 Text Sentiment Analysis")
    
    # Dynamically calculate the model path
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "models", "bert_sentiment"))

    # Check if the path exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at path: {MODEL_PATH}")
    else:
        # Load the model and tokenizer
        tokenizer, model = load_text_model(MODEL_PATH)

        # User input with unique key and placeholder
        user_input = st.text_area(
            "🖋️ Enter your text:",
            height=150,
            key="text_sentiment_input",
            placeholder="Type or paste your text here..."
        )

        # Button to analyze sentiment with tooltip
        col1, col2 = st.columns([4, 1])
        with col1:
            analyze_btn = st.button("🔍 Analyze Text Sentiment", key="analyze_text_sentiment_button")
        with col2:
            st.markdown(
                """
                <div class="tooltip">
                    <i class="fas fa-info-circle"></i>
                    <span class="tooltiptext">Click to analyze the sentiment of your text</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if analyze_btn:
            if user_input.strip():
                with st.spinner("🔄 Analyzing sentiment..."):
                    try:
                        predicted_class, probabilities = analyze_sentiment(tokenizer, model, user_input)
                        # Assuming class 0 is negative and 1 is positive
                        sentiment_label = "Positive 😊" if predicted_class == 1 else "Negative 😞"
                        sentiment_score = probabilities[0][1] if predicted_class == 1 else probabilities[0][0]
                    except Exception as e:
                        st.error(f"❌ An error occurred during sentiment analysis: {e}")
                        sentiment_label = "N/A"
                        sentiment_score = 0.0

                # Display the results
                st.markdown("---")
                st.subheader("🎯 Text Analysis Result")
                st.markdown(
                    f"<div class='results'><strong>Sentiment:</strong> {sentiment_label}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(sentiment_score)
                st.markdown(
                    f"<div class='results'><strong>Positive:</strong> {probabilities[0][1]:.4f}<br>"
                    f"<strong>Negative:</strong> {probabilities[0][0]:.4f}</div>",
                    unsafe_allow_html=True,
                )
                
                # Create a pie chart
                sentiment_data = {
                    'Sentiment': ['Positive 😊', 'Negative 😞'],
                    'Probability': [probabilities[0][1], probabilities[0][0]]
                }
                fig_pie = px.pie(
                    sentiment_data, 
                    names='Sentiment', 
                    values='Probability', 
                    title='Sentiment Distribution',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive 😊': '#2ecc71',
                        'Negative 😞': '#e74c3c'
                    }
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                center_plotly_chart(fig_pie, width=600, height=400)  # Center the chart
                
                # Create a bar chart
                fig_bar = px.bar(
                    sentiment_data, 
                    x='Sentiment', 
                    y='Probability',
                    title='Sentiment Scores',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive 😊': '#2ecc71',
                        'Negative 😞': '#e74c3c'
                    },
                    text='Probability'
                )
                fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                fig_bar.update_layout(yaxis=dict(range=[0,1]))
                center_plotly_chart(fig_bar, width=600, height=400)  # Center the chart

                # Option to download the results
                st.markdown(download_results(sentiment_data, "text_sentiment_results.csv"), unsafe_allow_html=True)
            else:
                st.warning("🚨 Please enter some text before analyzing!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# IMAGE EMOTION ANALYSIS
# ------------------------------
with tab2:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("🖼️ Image Emotion Analysis")

    # Image upload with unique key and tooltip
    uploaded_image = st.file_uploader(
        "📁 Upload an image for emotion analysis",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
        help="Supported formats: jpg, jpeg, png, webp"
    )

    # Button to analyze image emotion with tooltip
    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_image_btn = st.button("🔍 Analyze Image Emotion", key="analyze_image_emotion_button")
    with col2:
        st.markdown(
            """
            <div class="tooltip">
                <i class="fas fa-info-circle"></i>
                <span class="tooltiptext">Click to analyze the emotions in the uploaded image</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_image_btn:
        if uploaded_image is not None:
            with st.spinner("🔄 Analyzing emotions..."):
                try:
                    # Display the uploaded image with hover effect
                    image = Image.open(uploaded_image).convert("RGB")
                    # Convert image to base64 to embed directly
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as buffered:
                        image.save(buffered, format="JPEG")
                        img_str = get_base64_of_bin_file(buffered.name)
                    
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="data:image/jpeg;base64,{img_str}" class="hover-image" style="width:100%; max-width:500px;" alt="Uploaded Image">
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    # Send the image to the backend
                    with open(buffered.name, "rb") as img_file:
                        response = send_request(
                            "http://127.0.0.1:8000/analyze-image/",
                            files={"file": img_file}
                        )
                    
                    # Remove the temporary file
                    os.remove(buffered.name)
                except Exception as e:
                    st.error(f"❌ An error occurred while processing the image: {e}")
                    response = None

            # Handle the response
            if response and response.status_code == 200:
                try:
                    # Wrap the backend response in an expander
                    with st.expander("🔍 **Backend Response:**", expanded=False):
                        st.json(response.json())
                    
                    result = response.json()

                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.markdown("---")
                        st.subheader("🎯 Image Analysis Result")
                        st.markdown(
                            f"<div class='results'><strong>Emotion:</strong> {result['emotion']} {get_emotion_emoji(result['emotion'])}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='results'><strong>Score:</strong> {result['score']:.2f}</div>",
                            unsafe_allow_html=True,
                        )
                        
                        # Check if 'emotions_distribution' is present
                        emotions = result.get('emotions_distribution')
                        if emotions:
                            emotion_data = {
                                'Emotion': list(emotions.keys()),
                                'Score': list(emotions.values())
                            }
                        else:
                            # If 'emotions_distribution' is not present, create a distribution based on the dominant emotion
                            dominant_emotion = result['emotion']
                            emotion_data = {
                                'Emotion': [dominant_emotion],
                                'Score': [result['score']]
                            }
                        
                        # Create a bar chart for emotions
                        fig_emotions = px.bar(
                            emotion_data, 
                            x='Emotion', 
                            y='Score',
                            title='Emotion Distribution',
                            color='Emotion',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            text='Score'
                        )
                        fig_emotions.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                        fig_emotions.update_layout(yaxis=dict(range=[0,1]))
                        center_plotly_chart(fig_emotions, width=600, height=400)  # Center the chart

                        # Option to download the results
                        st.markdown(download_results(emotion_data, "image_emotion_results.csv"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Failed to parse the response: {e}")
            else:
                if response:
                    try:
                        error_message = response.json().get('detail', 'Unknown error')
                    except:
                        error_message = response.text
                    st.error(f"❌ Error: {error_message}")
        else:
            st.warning("🚨 Please upload an image before analyzing!")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# AUDIO EMOTION & SENTIMENT ANALYSIS
# ------------------------------
with tab3:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("🔊 Audio Emotion & Sentiment Analysis")

    # Audio upload with unique key and tooltip
    uploaded_audio = st.file_uploader(
        "📁 Upload an audio file for emotion and sentiment analysis",
        type=["mp3", "wav", "ogg", "flac"],
        key="audio_upload",
        help="Supported formats: mp3, wav, ogg, flac"
    )

    # Button to analyze audio with tooltip
    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_audio_btn = st.button("🔍 Analyze Audio Emotion & Sentiment", key="analyze_audio_emotion_sentiment_button")
    with col2:
        st.markdown(
            """
            <div class="tooltip">
                <i class="fas fa-info-circle"></i>
                <span class="tooltiptext">Click to analyze emotions and sentiment in the uploaded audio</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_audio_btn:
        if uploaded_audio is not None:
            with st.spinner("🔄 Analyzing audio..."):
                try:
                    # Display the audio player
                    st.audio(uploaded_audio, format=uploaded_audio.type)

                    # Use tempfile to handle temporary files
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as temp_audio:
                        temp_audio.write(uploaded_audio.read())
                        temp_audio_path = temp_audio.name

                    # Send the audio to the backend
                    with open(temp_audio_path, "rb") as audio_file:
                        response = send_request(
                            "http://127.0.0.1:8000/analyze-audio/",
                            files={"file": audio_file}
                        )

                    # Remove the temporary file
                    os.remove(temp_audio_path)
                except Exception as e:
                    st.error(f"❌ An error occurred while processing the audio: {e}")
                    response = None

            # Handle the response
            if response and response.status_code == 200:
                try:
                    # Wrap the backend response in an expander
                    with st.expander("🔍 **Backend Response:**", expanded=False):
                        st.json(response.json())
                    
                    result = response.json()

                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.markdown("---")
                        st.subheader("🎯 Audio Analysis Result")
                        
                        # Display Emotion
                        emotion = result['audio_emotion']['emotion']
                        emotion_score = result['audio_emotion']['score']
                        emotion_emoji = get_emotion_emoji(emotion)
                        st.markdown(
                            f"<div class='results'><strong>Audio Emotion:</strong> {emotion.capitalize()} {emotion_emoji}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='results'><strong>Emotion Score:</strong> {emotion_score:.2f}</div>",
                            unsafe_allow_html=True,
                        )
                        
                        # Display Transcript
                        transcript = result.get('transcript', 'N/A')
                        st.markdown(
                            f"<div class='results'><strong>Transcript:</strong> {transcript}</div>",
                            unsafe_allow_html=True,
                        )
                        
                        # Display Sentiment
                        sentiment = result['text_sentiment']['sentiment']
                        sentiment_score = result['text_sentiment']['score']
                        sentiment_emoji = "😊" if sentiment.lower() == "positive" else "😞" if sentiment.lower() == "negative" else "😐"
                        st.markdown(
                            f"<div class='results'><strong>Text Sentiment:</strong> {sentiment.capitalize()} {sentiment_emoji}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='results'><strong>Sentiment Score:</strong> {sentiment_score:.2f}</div>",
                            unsafe_allow_html=True,
                        )
                        
                        # Pie chart for sentiment
                        sentiment_data = {
                            'Sentiment': ['Positive 😊', 'Negative 😞'],
                            'Probability': [sentiment_score, 1 - sentiment_score]
                        }
                        fig_pie_audio = px.pie(
                            sentiment_data, 
                            names='Sentiment', 
                            values='Probability', 
                            title='Sentiment Distribution',
                            color='Sentiment',
                            color_discrete_map={
                                'Positive 😊': '#2ecc71',
                                'Negative 😞': '#e74c3c'
                            }
                        )
                        fig_pie_audio.update_traces(textposition='inside', textinfo='percent+label')
                        center_plotly_chart(fig_pie_audio, width=600, height=400)  # Center the chart
                        
                        # Bar chart for vocal emotions
                        emotions_audio = result.get('audio_emotions_distribution')
                        if emotions_audio:
                            emotion_audio_data = {
                                'Emotion': list(emotions_audio.keys()),
                                'Score': list(emotions_audio.values())
                            }
                        else:
                            # If 'audio_emotions_distribution' is not present, create a distribution based on the dominant emotion
                            dominant_emotion = result['audio_emotion']['emotion']
                            emotion_audio_data = {
                                'Emotion': [dominant_emotion],
                                'Score': [result['audio_emotion']['score']]
                            }
                        
                        fig_emotions_audio = px.bar(
                            emotion_audio_data, 
                            x='Emotion', 
                            y='Score',
                            title='Audio Emotion Distribution',
                            color='Emotion',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            text='Score'
                        )
                        fig_emotions_audio.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                        fig_emotions_audio.update_layout(yaxis=dict(range=[0,1]))
                        center_plotly_chart(fig_emotions_audio, width=600, height=400)  # Center the chart

                        # Option to download the results
                        st.markdown(download_results(sentiment_data, "audio_sentiment_results.csv"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Failed to parse the response: {e}")
            else:
                if response:
                    try:
                        error_message = response.json().get('detail', 'Unknown error')
                    except:
                        error_message = response.text
                    st.error(f"❌ Error: {error_message}")
        else:
            st.warning("🚨 Please upload an audio file before analyzing!")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# MULTIMODAL ANALYSIS
# ------------------------------
with tab4:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("🔗 Multimodal Analysis")

    st.markdown(
        """
        <p class='subtitle'>Combine text, image, and audio inputs for a comprehensive sentiment and emotion analysis.</p>
        """,
        unsafe_allow_html=True,
    )

    # User inputs with unique keys and placeholders
    user_text_mm = st.text_area(
        "🖋️ Enter your text:",
        height=150,
        key="multimodal_text_input",
        placeholder="Type or paste your text here..."
    )
    uploaded_image_mm = st.file_uploader(
        "📁 Upload an image for emotion analysis",
        type=["jpg", "jpeg", "png", "webp"],
        key="multimodal_image_upload",
        help="Supported formats: jpg, jpeg, png, webp"
    )
    uploaded_audio_mm = st.file_uploader(
        "📁 Upload an audio file for emotion and sentiment analysis",
        type=["mp3", "wav", "ogg", "flac"],
        key="multimodal_audio_upload",
        help="Supported formats: mp3, wav, ogg, flac"
    )

    # Button to analyze multimodal input with tooltip
    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_multimodal_btn = st.button("🔍 Analyze Multimodal", key="analyze_multimodal_button")
    with col2:
        st.markdown(
            """
            <div class="tooltip">
                <i class="fas fa-info-circle"></i>
                <span class="tooltiptext">Click to analyze combined text, image, and audio inputs</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_multimodal_btn:
        if not (user_text_mm.strip() or uploaded_image_mm or uploaded_audio_mm):
            st.warning("🚨 Please provide at least one input (text, image, or audio) for analysis.")
        else:
            with st.spinner("🔄 Analyzing multimodal inputs..."):
                # Prepare the data
                files = {}
                data = {}
                
                try:
                    if uploaded_image_mm:
                        # Save the image temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
                            image = Image.open(uploaded_image_mm).convert("RGB")
                            image.save(temp_image, format="JPEG")
                            temp_image_path = temp_image.name
                            files["image"] = open(temp_image_path, "rb")
                    
                    if uploaded_audio_mm:
                        # Save the audio temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio_mm.name)[1]) as temp_audio:
                            temp_audio.write(uploaded_audio_mm.read())
                            temp_audio_path = temp_audio.name
                            files["audio"] = open(temp_audio_path, "rb")
                    
                    if user_text_mm.strip():
                        data["text"] = user_text_mm.strip()
                    
                    # Send the multimodal request
                    response = send_request(
                        "http://127.0.0.1:8000/analyze-multimodal/",
                        data=data if data else None,
                        files=files if files else None,
                        method='POST'
                    )
                except Exception as e:
                    st.error(f"❌ An error occurred while preparing the inputs: {e}")
                    response = None
                finally:
                    # Close and remove temporary files
                    for file in files.values():
                        try:
                            file.close()
                            os.remove(file.name)
                        except Exception as e:
                            st.warning(f"⚠️ Could not remove temporary file: {e}")

            # Handle the response
            if response and response.status_code == 200:
                try:
                    # Wrap the backend response in an expander
                    with st.expander("🔍 **Backend Response:**", expanded=False):
                        st.json(response.json())
                    
                    result = response.json()

                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.markdown("---")
                        st.subheader("🎯 Multimodal Analysis Result")
                        
                        # Display individual results
                        individual = result.get("individual_results", {})
                        
                        sentiment_scores = []
                        emotions_scores = []
                        
                        # Text Sentiment
                        if "text_sentiment" in individual:
                            sentiment = individual['text_sentiment']['sentiment']
                            sentiment_score = individual['text_sentiment']['score']
                            sentiment_emoji = "😊" if sentiment.lower() == "positive" else "😞" if sentiment.lower() == "negative" else "😐"
                            st.markdown(
                                f"<div class='results'><strong>Text Sentiment:</strong> {sentiment.capitalize()} {sentiment_emoji}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='results'><strong>Sentiment Score:</strong> {sentiment_score:.2f}</div>",
                                unsafe_allow_html=True,
                            )
                            sentiment_scores.append({
                                'Mode': 'Text',
                                'Sentiment': f"{sentiment.capitalize()} {sentiment_emoji}",
                                'Score': sentiment_score
                            })
                        
                        # Image Emotion
                        if "image_emotion" in individual:
                            emotion = individual['image_emotion']['emotion']
                            score = individual['image_emotion']['score']
                            emotion_emoji = get_emotion_emoji(emotion)
                            st.markdown(
                                f"<div class='results'><strong>Image Emotion:</strong> {emotion.capitalize()} {emotion_emoji}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='results'><strong>Emotion Score:</strong> {score:.2f}</div>",
                                unsafe_allow_html=True,
                            )
                            emotions_scores.append({
                                'Mode': 'Image',
                                'Emotion': f"{emotion.capitalize()} {emotion_emoji}",
                                'Score': score
                            })
                        
                        # Audio Emotion
                        if "audio_emotion" in individual:
                            emotion = individual['audio_emotion']['emotion']
                            score = individual['audio_emotion']['score']
                            emotion_emoji = get_emotion_emoji(emotion)
                            st.markdown(
                                f"<div class='results'><strong>Audio Emotion:</strong> {emotion.capitalize()} {emotion_emoji}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='results'><strong>Emotion Score:</strong> {score:.2f}</div>",
                                unsafe_allow_html=True,
                            )
                            emotions_scores.append({
                                'Mode': 'Audio',
                                'Emotion': f"{emotion.capitalize()} {emotion_emoji}",
                                'Score': score
                            })
                        
                        # Audio Text Sentiment
                        if "audio_text_sentiment" in individual:
                            sentiment = individual['audio_text_sentiment']['sentiment']
                            sentiment_score = individual['audio_text_sentiment']['score']
                            sentiment_emoji = "😊" if sentiment.lower() == "positive" else "😞" if sentiment.lower() == "negative" else "😐"
                            transcript = individual.get('audio_transcript', 'N/A')  # Retrieve the transcript
                            st.markdown(
                                f"<div class='results'><strong>Audio Text Sentiment:</strong> {sentiment.capitalize()} {sentiment_emoji}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='results'><strong>Sentiment Score:</strong> {sentiment_score:.2f}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='results'><strong>Transcript:</strong> {transcript}</div>",
                                unsafe_allow_html=True,
                            )
                            sentiment_scores.append({
                                'Mode': 'Audio (Text)',
                                'Sentiment': f"{sentiment.capitalize()} {sentiment_emoji}",
                                'Score': sentiment_score
                            })
                        
                        # Display Weights
                        weights = result.get("weights", {})
                        if weights:
                            st.markdown("---")
                            st.subheader("📊 Assigned Weights")
                            weights_df = pd.DataFrame({
                                'Mode': ['Text', 'Image', 'Audio'],
                                'Weight': [weights.get('text_weight', 0.0), weights.get('image_weight', 0.0), weights.get('audio_weight', 0.0)]
                            })
                            st.table(weights_df)

                            # Bar chart for weights
                            fig_weights = px.bar(
                                weights_df,
                                x='Mode',
                                y='Weight',
                                title='Assigned Weights to Each Mode',
                                color='Mode',
                                color_discrete_map={
                                    'Text': '#3498db',
                                    'Image': '#e74c3c',
                                    'Audio': '#2ecc71'
                                },
                                text='Weight',
                                width=600,  # Set the width
                                height=400  # Optional: set the height if needed
                            )
                            fig_weights.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                            fig_weights.update_layout(yaxis=dict(range=[0,1]))
                            center_plotly_chart(fig_weights, width=600, height=400)  # Center the chart
                        
                        # Display Fused Results
                        fused = result.get("fused_results", {})
                        if fused:
                            st.markdown("---")
                            st.subheader("🌟 Fused Analysis Results")
                            
                            # Overall Sentiment
                            overall_sentiment = fused.get('overall_sentiment', {})
                            sentiment = overall_sentiment.get('sentiment', 'N/A')
                            sentiment_score = overall_sentiment.get('score', 0.0)
                            sentiment_emoji = "😊" if sentiment.lower() == "positive" else "😞" if sentiment.lower() == "negative" else "😐"
                            st.markdown(
                                f"<div class='results'><strong>Overall Sentiment:</strong> {sentiment} {sentiment_emoji}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='results'><strong>Sentiment Score:</strong> {sentiment_score:.2f}</div>",
                                unsafe_allow_html=True,
                            )

                            # Overall Emotion
                            overall_emotion = fused.get('overall_emotion', {})
                            emotion = overall_emotion.get('emotion', 'N/A')
                            emotion_score = overall_emotion.get('score', 0.0)
                            emotion_emoji = get_emotion_emoji(emotion)
                            st.markdown(
                                f"<div class='results'><strong>Overall Emotion:</strong> {emotion} {emotion_emoji}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='results'><strong>Emotion Score:</strong> {emotion_score:.2f}</div>",
                                unsafe_allow_html=True,
                            )

                            # Create a 2x2 grid for charts
                            with st.container():
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Pie chart for fused sentiment
                                    fused_sentiment_data = {
                                        'Sentiment': ['Positive 😊', 'Negative 😞'],
                                        'Probability': [
                                            fused.get('overall_sentiment', {}).get('score', 0.0),
                                            1 - fused.get('overall_sentiment', {}).get('score', 0.0)
                                        ]
                                    }
                                    fig_pie_fused_sentiment = px.pie(
                                        fused_sentiment_data, 
                                        names='Sentiment', 
                                        values='Probability',
                                        title='Fused Sentiment Distribution',
                                        color='Sentiment',
                                        color_discrete_map={
                                            'Positive 😊': '#2ecc71',
                                            'Negative 😞': '#e74c3c'
                                        }
                                    )
                                    fig_pie_fused_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                                    center_plotly_chart(fig_pie_fused_sentiment, width=600, height=400)  # Center the chart

                                with col2:
                                    # Bar chart for fused emotions
                                    fused_emotion_scores = {}
                                    # Aggregate individual emotions based on weights
                                    if "image_emotion" in individual:
                                        emo = individual['image_emotion']['emotion']
                                        score = individual['image_emotion']['score']
                                        weight = weights.get('image_weight', 0.0)
                                        if emo in fused_emotion_scores:
                                            fused_emotion_scores[emo] += score * weight
                                        else:
                                            fused_emotion_scores[emo] = score * weight
                                    if "audio_emotion" in individual:
                                        emo = individual['audio_emotion']['emotion']
                                        score = individual['audio_emotion']['score']
                                        weight = weights.get('audio_weight', 0.0)
                                        if emo in fused_emotion_scores:
                                            fused_emotion_scores[emo] += score * weight
                                        else:
                                            fused_emotion_scores[emo] = score * weight
                                    
                                    # Normalize emotions
                                    total_emotion_score = sum(fused_emotion_scores.values())
                                    if total_emotion_score > 0:
                                        for emo in fused_emotion_scores:
                                            fused_emotion_scores[emo] /= total_emotion_score
                                    
                                    # Create DataFrame for the chart
                                    if fused_emotion_scores:
                                        fused_emotion_data = {
                                            'Emotion': list(fused_emotion_scores.keys()),
                                            'Score': list(fused_emotion_scores.values())
                                        }
                                        fig_fused_emotions = px.bar(
                                            fused_emotion_data, 
                                            x='Emotion', 
                                            y='Score',
                                            title='Overall Emotion Distribution',
                                            color='Emotion',
                                            color_discrete_sequence=px.colors.qualitative.Set3,
                                            text='Score'
                                        )
                                        fig_fused_emotions.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                                        fig_fused_emotions.update_layout(yaxis=dict(range=[0,1]))
                                        center_plotly_chart(fig_fused_emotions, width=600, height=400)  # Center the chart
                        
                        # Option to download the results
                        st.markdown(download_results(result, "multimodal_analysis_results.json"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Failed to parse the response: {e}")
            else:
                if response:
                    try:
                        error_message = response.json().get('detail', 'Unknown error')
                    except:
                        error_message = response.text
                    st.error(f"❌ Error: {error_message}")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# TWITTER SENTIMENT ANALYSIS
# ------------------------------
with tab5:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("🐦 Twitter Sentiment Analysis")
    st.markdown(
        """
        <p class='subtitle'>Analyze the sentiment of tweets based on a keyword or hashtag. You can choose to use real Twitter data or mock data for testing purposes.</p>
        """,
        unsafe_allow_html=True,
    )
    query = st.text_input(
        "🔍 Enter a keyword or hashtag for Twitter analysis:",
        key="twitter_query_input",
        placeholder="#OpenAI or OpenAI"
    )
    max_tweets = st.slider(
        "📈 Number of tweets to fetch:",
        10,
        500,
        100,
        key="twitter_max_tweets_slider"
    )
    use_mock = st.checkbox(
        "🧪 Use mock data instead of real Twitter data",
        key="twitter_use_mock_checkbox"
    )

    # Button to analyze tweets with tooltip
    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_twitter_btn = st.button("🔍 Analyze Twitter Sentiment", key="analyze_twitter_sentiment_button_unique")
    with col2:
        st.markdown(
            """
            <div class="tooltip">
                <i class="fas fa-info-circle"></i>
                <span class="tooltiptext">Click to fetch and analyze tweets based on your query</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_twitter_btn:
        if query.strip():
            with st.spinner("🔄 Fetching and analyzing tweets..."):
                try:
                    endpoint = "http://127.0.0.1:8000/analyze-twitter/"
                    params = {"query": query, "max_tweets": max_tweets, "use_mock": use_mock}
                    response = send_request(endpoint, data=params, method='GET')
                except Exception as e:
                    st.error(f"❌ An error occurred while sending the request: {e}")
                    response = None

            # Handle the response
            if response and response.status_code == 200:
                try:
                    # Wrap the backend response in an expander
                    with st.expander("🔍 **Backend Response:**", expanded=False):
                        st.json(response.json())
                    
                    result = response.json()

                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.markdown("---")
                        st.subheader("🎯 Twitter Analysis Result")
                        
                        sentiment_distribution = result.get("sentiment_distribution", {})
                        sentiments = result.get("sentiments", [])

                        # Display Results
                        st.markdown(f"**Query:** `{query}`")
                        st.markdown(f"**Positive Tweets:** {sentiment_distribution.get('positive', 0) * 100:.2f}%")
                        st.markdown(f"**Negative Tweets:** {sentiment_distribution.get('negative', 0) * 100:.2f}%")

                        # Create distribution charts
                        distribution_data = {
                            'Sentiment': ['Positive 😊', 'Negative 😞'],
                            'Percentage': [sentiment_distribution.get('positive', 0) * 100, sentiment_distribution.get('negative', 0) * 100]
                        }
                        fig_distribution = px.bar(
                            distribution_data,
                            x='Sentiment',
                            y='Percentage',
                            title='Sentiment Distribution',
                            color='Sentiment',
                            color_discrete_map={
                                'Positive 😊': '#2ecc71',
                                'Negative 😞': '#e74c3c'
                            },
                            text='Percentage'
                        )
                        fig_distribution.update_traces(texttemplate='%{text:.2f}%', textposition='auto')
                        fig_distribution.update_layout(yaxis=dict(range=[0, 100]))
                        center_plotly_chart(fig_distribution, width=600, height=400)

                        # Show Detailed Sentiments
                        st.write("### Detailed Sentiments")
                        if sentiments:
                            for idx, tweet in enumerate(sentiments, 1):
                                # Check for keys present in the response
                                tweet_text = tweet.get('tweet', tweet.get('text', 'N/A'))
                                sentiment = tweet.get('sentiment', 'N/A')
                                sentiment_score = tweet.get('sentiment_score', None)
                                media_analysis = tweet.get('media_analysis', [])

                                # Use an expander for each tweet with a progressive number
                                with st.expander(f"Tweet {idx}: {sentiment.capitalize()} - Score: {sentiment_score:.2f}" if sentiment_score else f"Tweet {idx}: {sentiment.capitalize()}"):
                                    # Display tweet text and sentiment
                                    if sentiment_score is not None:
                                        st.markdown(f"**Tweet:** {tweet_text}")
                                        st.markdown(f"**Sentiment Score:** {sentiment_score:.2f}")
                                    else:
                                        st.markdown(f"**Tweet:** {tweet_text}")
                                        st.markdown(f"**Sentiment Score:** N/A")

                                    # Display media analysis (if present)
                                    if media_analysis:
                                        st.markdown("**Media Analysis:**")
                                        for media in media_analysis:
                                            media_type = media.get('type', 'unknown')
                                            media_url = media.get('url', 'N/A')
                                            emotion = media.get('emotion', 'N/A')
                                            emotion_score = media.get('emotion_score', 'N/A')

                                            # If the media is an image, display it
                                            if media_type == 'image' and media_url.startswith("http"):
                                                st.image(media_url, caption=f"Emotion: {emotion.capitalize()} (Score: {emotion_score})", width=300)
                                            elif media_type == 'audio':
                                                st.markdown(f"🎵 **Audio URL:** [Listen here]({media_url})")
                                                st.markdown(f" - **Emotion:** {emotion.capitalize()} (Score: {emotion_score})")
                                            else:
                                                st.markdown(f" - **{media_type.capitalize()} URL:** {media_url}")
                                                st.markdown(f" - **Emotion:** {emotion.capitalize()} (Score: {emotion_score})")
                        else:
                            st.warning("No tweets found matching the query.")

                        # Option to download the results
                        st.markdown(download_results(result, "twitter_sentiment_results.json"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Failed to parse the response: {e}")
            else:
                if response:
                    try:
                        error_message = response.json().get('detail', 'Unknown error')
                    except:
                        error_message = response.text
                    st.error(f"❌ Error: {error_message}")
        else:
            st.warning("🚨 Please enter a keyword or hashtag!")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    """
    <footer>
        <p>© 2024 Sentiment & Emotion Analysis App</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
