from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
from io import BytesIO


class ImageEmbeddingEngine():
    def __init__(self,image_urls, model_name):
        # Initialize the model and processor once
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_urls=image_urls

    def __get_image_embedding(self, image_path):
        """This method is for internal use and not exposed outside the class."""
        image_bytes = requests.get(image_path).content
        image = Image.open(BytesIO(image_bytes))
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs).numpy()
        return embedding.squeeze()

    def get_embeddings_for_images(self):
        """Public method to get embeddings for a list of image URLs."""
        embeddings = []
        for image in self.image_urls:
            image_embedding = self.__get_image_embedding(image)
            embeddings.append(image_embedding)
        return embeddings, self.image_urls