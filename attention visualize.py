import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load a pre-trained sentiment model and tokenizer
# We must set output_attentions=True to access the matrices
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)

# 2. Prepare the input
text = "The battery life is amazing, but the screen is dim."
inputs = tokenizer(text, return_tensors="pt")
print("123", type(inputs), inputs)

# 3. Run the model
outputs = model(**inputs)
logits = outputs.logits
attentions = outputs.attentions  # This is a tuple of 6 layers (for DistilBERT)
print("123", type(attentions), attentions)

# 4. Get the prediction
prediction = torch.argmax(logits, dim=-1)
sentiment = "POSITIVE" if prediction.item() == 1 else "NEGATIVE"
print(f"Predicted Sentiment: {sentiment}")

# 5. Extract attention from the last layer (Layer 6)
# Shape: [batch_size, num_heads, sequence_length, sequence_length]
last_layer_attention = attentions[-1][0]

# Average across all 12 attention heads to get a general overview
avg_attention = last_layer_attention.mean(dim=0)

# 6. Display which words the first word ([CLS] token) "attended" to most
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
cls_attention = avg_attention[0] # Focus on the first token which represents the whole sentence

print("\nWord Importance (Attention from [CLS] token):")
for token, score in zip(tokens, cls_attention):
    print(f"{token:10} : {score.item():.4f}")

