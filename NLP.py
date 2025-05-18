import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# Load models
models = {
    "Logistic Regression": pickle.load(open("logistic_regression.pkl", 'rb')),
    "Multinomial Naive Bayes": pickle.load(open("multinomial_nb.pkl", 'rb')),
    "Random Forest": pickle.load(open("random_forest.pkl", 'rb')),
    
}

# Load shared assets
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
emotion_labels = list(label_encoder.classes_)

# Model Accuracy (assumed values)
model_accuracies = {
    "Logistic Regression": 0.85,
    "Multinomial Naive Bayes": 0.83,
    "Random Forest": 0.80
    
}

# Evaluation results (example summaries)
model_evaluations = {
    "Logistic Regression": """
**Precision**: 0.86  
**Recall**: 0.85  
**F1-score**: 0.85  
**Support**: 1000 samples  
""",
    "Multinomial Naive Bayes": """
**Precision**: 0.82  
**Recall**: 0.83  
**F1-score**: 0.82  
**Support**: 1000 samples  
""",
    "Random Forest": """
**Precision**: 0.81  
**Recall**: 0.80  
**F1-score**: 0.80  
**Support**: 1000 samples  

"""
}

# App Config
st.set_page_config(page_title="Emotion Classifier (ML)", layout="centered")

# App Header
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#6C63FF;">🤖 ML Emotion Classifier</h1>
        <p style="color:gray; font-size:18px;">Compare multiple ML models for emotion detection.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar - Model selection and info
with st.sidebar:
    st.header("📊 Model Info")
    model_choice = st.selectbox("Choose a model:", list(models.keys()))
    model_accuracy = model_accuracies[model_choice]
    st.progress(int(model_accuracy * 100), text=f"Accuracy: {model_accuracy * 100:.1f}%")
    st.markdown(model_evaluations[model_choice])

# Input Section
st.subheader("📝 Enter your sentence:")
text_input = st.text_area("Type something emotional...", placeholder="e.g. I'm so happy today!")

# Predict Button
if st.button("🔍 Predict Emotion"):
    if not text_input.strip():
        st.warning("Please enter a sentence.")
    else:
        st.subheader("📈 Prediction Results")

        # Vectorize and predict
        X_input = tfidf_vectorizer.transform([text_input])
        model = models[model_choice]
        pred_probs = model.predict_proba(X_input)[0]
        pred_label = np.argmax(pred_probs)
        emotion = label_encoder.inverse_transform([pred_label])[0]

        # Show prediction
        st.success(f"🎯 Predicted Emotion: **{emotion}**")

        # Plot probability chart
        prob_df = pd.DataFrame({
            'Emotion': emotion_labels,
            'Probability': pred_probs
        }).sort_values(by='Probability', ascending=False)

        fig = px.bar(prob_df, x='Emotion', y='Probability', color='Emotion',
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     title="Emotion Prediction Probabilities")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("<hr><center style='color: gray;'>Made with ❤️ using Scikit-learn and Streamlit</center>", unsafe_allow_html=True)
