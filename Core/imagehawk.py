from Core.__Engines.image_embeddings import ImageEmbeddingEngine
from Core.__Engines.text_embeddings import TextEmbeddingEngine
from Core.__VectorStores.qdrantvectorstore import SaveEmbeddings
from qdrant_client import QdrantClient
import os
from typing import List

class ImageHawk():
    def __init__(self,
            new_collection:bool,
            image_urls:List[str],
            userid:str,
            project:str,
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY")
            ):
        
        self.new_collection=new_collection
        self.image_urls=image_urls
        self.qdrant_url=qdrant_url
        self.qdrant_api_key=qdrant_api_key
        self.userid=userid
        self.project=project

    def generate_image_embeddings(self):
        embeddings,image_urls=ImageEmbeddingEngine(self.image_urls)
        save_result=SaveEmbeddings(
            new=bool(self.new_collection),qdrant_url=self.qdrant_url,qdrant_api_key=self.qdrant_api_key,
            embeddings=embeddings,image_urls=image_urls,userid=self.userid,
            agent=self.project
        )
        if save_result==True:
            return True
        elif save_result==False:
            return False


    def search_similar_images(self,text):

        text_embedding =TextEmbeddingEngine(text)
        client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key,prefer_grpc=True)
        search_result = client.search(
        collection_name=f"{self.userid}_{self.project}_image",
        query_vector=text_embedding.tolist(),
        limit=1
        )

        #Display the search results
        for result in search_result:
            output={
                "id":result.id,
                "similarity_score":result.score,
                "metadata":result.payload
            }
            return output

