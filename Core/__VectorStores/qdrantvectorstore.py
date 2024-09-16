from qdrant_client.models import PointStruct,VectorParams
from qdrant_client import QdrantClient


def SaveEmbeddings(new,qdrant_url,qdrant_api_key,embeddings,image_urls,userid,project):
    try:
        points = [
            PointStruct(id=i+1, vector=embedding.tolist(), payload={"image_url": url})
            for i, (embedding, url) in enumerate(zip(embeddings, image_urls))
            ]

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key,prefer_grpc=True)
        if new== True:
            client.create_collection(
                collection_name=f"{userid}_{project}_image",
                vectors_config=VectorParams(size=512, distance="Cosine") 
            )
            client.upsert(
            collection_name=f"{userid}_{project}_image",
            points=points
            )
        else:
            client.upsert(
            collection_name=f"{userid}_{project}_image",
            points=points
            )
        return True
    except Exception as e:
        print(e)
        return False