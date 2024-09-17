from transformers import CLIPProcessor, CLIPModel
#from transformers import BlipProcessor, BlipForConditionalGeneration
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
        
        else:
            print("other model types not supported")


    def get_embeddings_for_images(self):
        """Public method to get embeddings for a list of image URLs."""
        embeddings = []
        for image in self.image_urls:
            image_embedding = self.__get_image_embedding(image)
            embeddings.append(image_embedding)
        return embeddings, self.image_urls
    


