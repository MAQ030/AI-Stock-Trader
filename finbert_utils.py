from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 

# Check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    """
    Estimate sentiment for a list of news headlines.

    Parameters:
    - news (list): List of news headlines.

    Returns:
    - Tuple[float, str]: Probability and sentiment label.
    """
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]

# Test the sentiment estimation function
if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
    print(tensor, sentiment)
    # Print whether CUDA (GPU support) is available
    print(torch.cuda.is_available())  # Comment added for clarity
