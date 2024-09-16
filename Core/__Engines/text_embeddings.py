from transformers import CLIPProcessor, CLIPModel
import torch

class TextEmbeddingEngine():
    def __init__(self,text,model_name):
        # Initialize the model and processor once
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.text=text

    def __get_text_embedding(self,text):
        """Private method to get the embedding for a single text."""
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs).numpy()
        return embedding.squeeze()

    def get_embedding_for_text(self):
        """Public method to expose the text embedding."""
        return self.__get_text_embedding(self.text)