from transformers import CLIPProcessor, CLIPModel
from transformers import FlavaFeatureExtractor, FlavaModel


from PIL import Image
import requests
import torch
from io import BytesIO


class ImageEmbeddingEngine():
    def __init__(self,image_urls,model_type):
        # Initialize the model and processor once
        self.image_urls=image_urls
        self.model_type=model_type
       

    def __get_image_embedding(self, image_path):
        """This method is for internal use and not exposed outside the class."""
        if self.model_type=="clip":

            image_bytes = requests.get(image_path).content
            image = Image.open(BytesIO(image_bytes))

            model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs).numpy()
            return embedding.squeeze()
        
        elif self.model_type=="flava":

            print("other model types not supported")
            model = FlavaModel.from_pretrained("facebook/flava-full")
            feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
            image = Image.open(requests.get(image_bytes, stream=True).raw)
            inputs = feature_extractor(images=[image], return_tensors="pt")
            image_embedding = model.get_image_features(**inputs)
            return image_embedding
        else:
            print("others models not supported yet")

    def get_embeddings_for_images(self):
        """Public method to get embeddings for a list of image URLs."""
        embeddings = []
        for image in self.image_urls:
            image_embedding = self.__get_image_embedding(image)
            embeddings.append(image_embedding)
        return embeddings, self.image_urls
    


