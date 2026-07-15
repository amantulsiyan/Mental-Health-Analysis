import os
import pandas as pd
import streamlit as st

from config import ASSETS_DIR
from vader_module import analyze_sentiment
from lime_explainer import explain_text


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mental Health Monitoring Dashboard",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🧠 Mental Health Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Analysis",
        "Corpus Overview",
        "Model Summary"
    ]
)


# --------------------------------------------------
# Helper Function
# --------------------------------------------------

def load_image(folder, filename):
    """
    Returns the absolute path of an image stored
    inside the assets folder.
    """
    return os.path.join(
        ASSETS_DIR,
        folder,
        filename
    )
# ==================================================
# ANALYSIS PAGE
# ==================================================

if page == "Analysis":

    st.title("🧠 Mental Health Text Analysis")

    st.write(
        "Enter a Reddit post below to analyse "
        "mental health, sentiment and explanations."
    )

    text = st.text_area(
        "Reddit Post",
        height=200,
        placeholder="Type or paste a Reddit post..."
    )

    if st.button("Analyze"):

        if text.strip() == "":

            st.warning("Please enter some text.")

        else:

            with st.spinner("Running RoBERTa + VADER + LIME..."):

                vader = analyze_sentiment(text)

                result = explain_text(text)

            st.success("Analysis Complete")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("🤖 RoBERTa Prediction")

                st.metric(
                    "Predicted Class",
                    result["prediction"].title()
                )

                st.metric(
                    "Confidence",
                    f"{result['confidence']*100:.2f}%"
                )

                st.metric(
                    "Inference Time",
                    f"{result['inference_time']} sec"
                )

            with col2:

                st.subheader("😊 VADER Sentiment")

                st.metric(
                    "Sentiment",
                    vader["label"]
                )

                st.metric(
                    "Compound Score",
                    round(vader["compound"], 3)
                )

            st.divider()
            st.subheader("📊 Class Probabilities")

            prob_df = pd.DataFrame(
                {
                    "Class": result["class_probabilities"].keys(),
                    "Probability": result["class_probabilities"].values()
                }      
            ).set_index("Class")
            st.bar_chart(prob_df)
            st.dataframe(prob_df)

            st.subheader("🔍 Important Words")
            for item in result["important_words"]:
                word = item["word"]
                score = item["importance"]
                if score > 0:
                    st.markdown(
                        f"<span style='color:green'><b>{word}</b></span> : {score:.3f}",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<span style='color:red'><b>{word}</b></span> : {score:.3f}",
                        unsafe_allow_html=True
                    )
            st.subheader("🖼️ LIME Explanation")
            if os.path.exists(result["png_path"]):
                st.image(
                    result["png_path"],
                    use_container_width=True
                )        


# ==================================================
# CORPUS OVERVIEW PAGE
# ==================================================

elif page == "Corpus Overview":
    st.title("📊 Corpus Overview")

    st.write(
        "Visual overview of the training dataset including "
        "word clouds, sentiment analysis and class distribution."
    )

    st.divider()
    st.subheader("☁️ Word Clouds")

    cols = st.columns(2)

    wordclouds = [
        ("Depression", "depression.png"),
        ("Anxiety", "anxiety.png"),
        ("Bipolar", "bipolar.png"),
        ("PTSD", "ptsd.png"),
        ("Normal", "normal.png")
    ]

    for i, (title, filename) in enumerate(wordclouds):

        with cols[i % 2]:

            st.markdown(f"### {title}")

            image_path = load_image(
                "wordclouds",
                filename
            )

            if os.path.exists(image_path):
                st.image(
                    image_path,
                    use_container_width=True
                )
            else:
                st.error(f"{filename} not found")

            st.divider()

    st.subheader("😊 Average Sentiment by Class")

    sentiment_chart = load_image(
        "charts",
        "sentiment_by_class.png"
    )

    if os.path.exists(sentiment_chart):
        st.image(
            sentiment_chart,
            use_container_width=True
        )
    else:
        st.warning("sentiment_by_class.png not found.")

    st.divider()

    st.subheader("📈 Class Distribution")

    distribution_chart = load_image(
        "charts",
        "class_distribution.png"
    )

    if os.path.exists(distribution_chart):
        st.image(
            distribution_chart,
            use_container_width=True
        )
    else:
        st.warning("class_distribution.png not found.")            



# ==================================================
# MODEL SUMMARY PAGE
# ==================================================

elif page == "Model Summary":

    st.title("📈 Model Summary")

    st.write(
        "Performance evaluation of the RoBERTa model."
    )

    st.divider()

    st.subheader("Confusion Matrix")

    confusion = load_image(
        "charts",
        "roberta_confusion_matrix.png"
    )

    if os.path.exists(confusion):
        st.image(
            confusion,
            use_container_width=True
        )
    else:
        st.warning("Confusion matrix not found.")

    st.divider()

    st.subheader("Training Curves")

    curves = load_image(
        "charts",
        "training_curves.png"
    )

    if os.path.exists(curves):
        st.image(
            curves,
            use_container_width=True
        )
    else:
        st.warning("Training curves not found.")

    st.divider()

    st.subheader("ROC Curves")

    roc = load_image(
        "charts",
        "roc_curves.png"
    )

    if os.path.exists(roc):
        st.image(
            roc,
            use_container_width=True
        )
    else:
        st.warning("ROC curves not found.")

    st.divider()

    st.subheader("Performance Comparison")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],
        "Baseline": [
            "84.11%",
            "84.64%",
            "83.99%",
            "84.12%"
        ],
        "RoBERTa": [
            "88.24%",
            "88.42%",
            "88.22%",
            "88.30%"
        ]
    })

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )