
### _ImageHawk_


`This package generates embeddings for the given photos and saves them in the qdrant vector store`
`It also generates embeddings for your text query`
`It provides  a function to find an image based on its semantic similarity to the text query`
`Currently, it supports the Qdrant Vector Store, but other stores will be added soon....`

#### _IMPORTANT_!

`Note: Before using the package , ensure your have follwing env variables "QDRANT_URL","QDRANT_API_KEY"`


#### _Usage_ 
```python
`Import`  
from griffin.Core import Griffin

`Class initialization`

_griffin=Griffin(
    model_type="clip",
    vectorstore="qdrant",
    collectionname="user:project"
)
`Image Embeddings  Generation and Saving in VectorStore`

_griffin.generate_image_embeddings(
    new_collection=True , # OR False if collection already exists,
    imageurls=["https://samplelink/to/image1.jpg","https://samplelink/to/image2.jpg"]
)
`Text to Image Similarity Search`

search_result=_griffin.search_similar_images(text="query text",result_limit=1)
print(search_result)

```


> Author:Junaid Bashir
> Note:Currently Qdrant is the only supported vectorstore (new stores coming soon)