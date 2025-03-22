import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

st.set_page_config(page_title="Hate Speech Detector", page_icon="🔍", layout="centered")


@st.cache_resource
def load_model():
    """Load the model and tokenizer"""
    import os
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = "model_even"
    # model = "model_odd"
    # model = "model_lora"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Verify model directory exists
        if not os.path.exists(model):
            raise Exception(f"{model} directory not found")

        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

        # Load model with specific configuration
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            local_files_only=True,
            num_labels=2,  # Binary classification
            problem_type="single_label_classification",
        )

        model = model.to(device)
        model.eval()

        return tokenizer, model, device

    except Exception as e:
        # Check what files exist in the directory
        files = os.listdir(model) if os.path.exists(model) else []
        raise Exception(f"Failed to load model: {str(e)}\nFiles in {model}: {files}")


def predict_toxicity(text, tokenizer, model, device):
    # Tokenize input
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding=True
    ).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
        confidence = probabilities[0][prediction.item()].item()

    return prediction.item(), confidence


def main():
    st.title("🔍 Hate Speech Detector")
    st.write("""
    This application helps detect potentially toxic or hate speech content in text.
    Enter your text below to analyze it.
    """)

    try:
        # Load model
        tokenizer, model, device = load_model()

        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="Type or paste your text here...",
        )

        if st.button("Analyze"):
            if text_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    # Get prediction
                    prediction, confidence = predict_toxicity(
                        text_input, tokenizer, model, device
                    )

                    # Display results
                    st.write("### Analysis Results")

                    if prediction == 1:
                        st.error(
                            f"⚠️ This text was classified as potentially toxic/hate speech (Confidence: {confidence:.2%})"
                        )
                    else:
                        st.success(
                            f"✅ This text appears to be non-toxic (Confidence: {confidence:.2%})"
                        )

                    # Display confidence meter
                    st.write("Confidence Level:")
                    st.progress(confidence)

    except Exception as e:
        st.error(f"""
        Error loading the model.
        
        Error: {str(e)}
        """)


if __name__ == "__main__":
    main()
