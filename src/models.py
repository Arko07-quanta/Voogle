import torch
from transformers import ASTModel

class AudioEmbeddingModel:
    """
    Uses State-of-the-Art Audio Spectrogram Transformer (AST) to generate general-purpose audio embeddings.
    """
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        print(f"Loading SOTA Embedding Model: {model_name}...")
        # We load the base model to get embeddings (hidden states), not the classification head
        self.model = ASTModel.from_pretrained(model_name)
        self.model.eval()
        
    def get_embeddings(self, input_values):
        """
        Passes the extracted features (spectrogram) through the AST model to get embeddings.
        """
        with torch.no_grad():
            # AST returns a tuple, the first element is the sequence of hidden states
            outputs = self.model(input_values)
            # We can take the pooled output (often the [CLS] token equivalent) as the general embedding
            embeddings = outputs.pooler_output 
        return embeddings

# Global instance to avoid reloading
_embedding_model = None

def get_general_purpose_embeddings(input_values):
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = AudioEmbeddingModel()
        
    return _embedding_model.get_embeddings(input_values)
