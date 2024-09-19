from Core.__Engines.image_embeddings import ImageEmbeddingEngine
from Core.__Engines.text_embeddings import TextEmbeddingEngine
from Core.__VectorStores.qdrantvectorstore import SaveEmbeddings
from qdrant_client import QdrantClient
import os



class Griffin():

    generated_and_saved_in_vector_store=False

    def __init__(self,
            model_type="clip",
            vectorstore="qdrant",
            collectionname="user:project"
            ):
        self.model_type=model_type
        self.vectorstore=vectorstore
        self.collectionname=collectionname.split(":")
        self.vectorstorecred={
            "qdrant":[os.getenv("QDRANT_URL"),os.getenv("QDRANT_API_KEY")]
        }

    def generate_image_embeddings(self,new_collection,imageurls):
        try:
        
            embeddings,image_urls=ImageEmbeddingEngine(imageurls,model_type=self.model_type)

            if self.vectorstore=="qdrant":
                relevant_cred=self.vectorstorecred.get("qdrant")
        
                save_result=SaveEmbeddings(
                        new=bool(new_collection),qdrant_url=relevant_cred[0],qdrant_api_key=relevant_cred[1],
                        embeddings=embeddings,image_urls=image_urls,userid=self.collectionname[0],project=self.collectionname[1]
                )
                if save_result==True:
                    #return True
                    Griffin.generated_and_saved_in_vector_store=True

                elif save_result==False:
                    Griffin.generated_and_saved_in_vector_store=False
            else:
                print("Qdrant Currently supported")

        except Exception as e:
             print(e)
             Griffin.generated_and_saved_in_vector_store=False

    def search_similar_images(self,text,result_limit):
            if Griffin.generated_and_saved_in_vector_store==True:

                text_embedding_engine =TextEmbeddingEngine(text,model_type=self.model_type)
                text_embedding=text_embedding_engine.get_embedding_for_text()

                if self.vectorstore=="qdrant":
                    relevant_cred=self.vectorstorecred.get("qdrant")
                    client = QdrantClient(url=relevant_cred[0], api_key=relevant_cred[1],prefer_grpc=True)
                    search_result = client.search(
                            collection_name=f"{self.collectionname[0]}_{self.collectionname[1]}_image",
                            query_vector=text_embedding.tolist(),
                            limit=int(result_limit)
                    )
                    for result in search_result:
                        output={
                        "id":result.id,
                        "similarity_score":result.score,
                        "metadata":result.payload
                        }
                        return output
            elif Griffin.generated_and_saved_in_vector_store==False:
                print("cannot call this method if embedding generation failed")    

    def image_to_image_similarity(self):
         pass
