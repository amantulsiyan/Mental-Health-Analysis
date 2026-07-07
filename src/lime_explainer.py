#Imports 
print("Setting up imports")
import os
import time
import torch
import numpy as np 
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from lime.lime_text import LimeTextExplainer
from config import *
from preprocess import roberta_clean

print("Imports added")

# ── Device Setup ─────────────────────────────────────────────

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"[2/10] Device selected: {device}")

# ── Load Model ───────────────────────────────────────────────

print("[3/10] Loading trained RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(MODEL_DIR, 'best_model')
)

model.to(device)
model.eval()

print("[4/10] Model loaded and moved to device successfully.")


# ── Load Tokenizer ───────────────────────────────────────────

print("[5/10] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_DIR, 'best_model')
)

print("[6/10] Tokenizer loaded successfully.")

# predict proba
def predict_proba(texts):

    #Preprocess texts
    texts = [roberta_clean(t) for t in texts]

    #Tokenise texts
    encoding = tokenizer(
        texts,
        padding=True,
        truncation = True,
        max_length = MAX_LEN,
        return_tensors = 'pt'
    )

    #Move tensors to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    #Disable gradients for inference
    with torch.no_grad():

        #Run model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        #Extract logits
        logits = outputs.logits

        #Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
    
    #Return NumPy probabilities
    return probs.cpu().numpy()


# create limetextexplainer
explainer = LimeTextExplainer(
    class_names = CLASSES,
    bow = True
)

# explain_text()
def explain_text(text, true_label=None):

    start = time.time()
    print("Generating LIME Explanation")
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
        top_labels=1
    )

    probs = predict_proba([text])

    pred_idx = np.argmax(probs[0])

    confidence = probs[0][pred_idx]

    predicted_class = CLASSES[pred_idx]

    important_words = exp.as_list(label = pred_idx)

    os.makedirs(ASSETS_DIR, exist_ok=True)

    html_dir = os.path.join(
        ASSETS_DIR,
        "lime_html"
    )

    os.makedirs(html_dir, exist_ok=True)

    explanation_path = os.path.join(
        html_dir,
        f"{predicted_class}_{int(time.time())}.html"
    )
    exp.save_to_file(explanation_path)

    lime_examples_dir = os.path.join(
        ASSETS_DIR,
        "lime_examples"
    )

    os.makedirs(lime_examples_dir, exist_ok=True)

    png_path = os.path.join(
        lime_examples_dir,
        f"{predicted_class}_{int(time.time())}.png"
    )

    fig = exp.as_pyplot_figure(label=pred_idx)

    plt.savefig(
        png_path,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    print("\nImportant words:")

    for word, score in important_words:
        print(f"{word}: {score:.4f}")

    print(f"\nExplanation saved to:\n{explanation_path}")

    class_probabilities = {

    CLASSES[i]: round(float(probs[0][i]), 4)

    for i in range(NUM_CLASSES)
    }
    end = time.time()

    inference_time = round(end - start, 2)
    return {

    "prediction": predicted_class,

    "true_label": true_label,

    "correct_prediction":
        predicted_class == true_label
        if true_label else None,

    "confidence": float(confidence),

    "important_words": [

    {
        "word": word,
        "importance": float(score)
    }

    for word, score in important_words
    ],

    "html_path": explanation_path,

    "png_path": png_path,

    "class_probabilities": class_probabilities,

    "input_text": text,

    "inference_time":inference_time
    }
if __name__ == "__main__":
    
    case_studies = [

    # ───────────── Depression ─────────────

    {
        "text": (
            "I feel completely hopeless and exhausted every day. "
            "Nothing excites me anymore and I barely get out of bed."
        ),
        "true_label": "depression"
    },

    {
        "text": (
            "Life feels meaningless lately and I have lost interest "
            "in everything I used to enjoy."
        ),
        "true_label": "depression"
    },


    # ───────────── Anxiety ─────────────

    {
        "text": (
            "I cannot stop worrying about every small thing and "
            "my chest feels tight all the time."
        ),
        "true_label": "anxiety"
    },

    {
        "text": (
            "My mind keeps racing at night and I constantly feel "
            "nervous about what could go wrong."
        ),
        "true_label": "anxiety"
    },


    # ───────────── Bipolar ─────────────

    {
        "text": (
            "Some days I feel unstoppable and full of energy, "
            "but other days I suddenly crash into sadness."
        ),
        "true_label": "bipolar"
    },

    {
        "text": (
            "I go through extreme mood swings where I feel overly "
            "confident one week and deeply depressed the next."
        ),
        "true_label": "bipolar"
    },


    # ───────────── PTSD ─────────────

    {
        "text": (
            "I still get nightmares and flashbacks from the accident "
            "and loud noises make me panic instantly."
        ),
        "true_label": "ptsd"
    },

    {
        "text": (
            "Ever since the traumatic incident, I avoid crowded places "
            "and constantly feel on edge."
        ),
        "true_label": "ptsd"
    },


    # ───────────── Normal ─────────────

    {
        "text": (
            "I had a productive and peaceful day today and enjoyed "
            "spending time with my friends."
        ),
        "true_label": "normal"
    },

    {
        "text": (
            "Work has been going well recently and I feel motivated "
            "and emotionally balanced."
        ),
        "true_label": "normal"
    }
]
    for sample in case_studies:
        result = explain_text(
            sample['text'],
            sample['true_label']
        )

        print("\n", '='*50, "\n")