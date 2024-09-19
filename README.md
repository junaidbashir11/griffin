
### _Griffin-Vision_


`A package for performing text-to-image and image-to-image searches using embedding-based techniques`

#### _IMPORTANT_!

` Note: Before using the package , ensure your have follwing env variables "QDRANT_URL","QDRANT_API_KEY" `

![griffin](./vulture.png)

#### _Usage_ 
```python
`Import`  
from griffin_vision.Core import Griffin

`Class initialization`

griffin_obj=Griffin(
    model_type="clip",
    vectorstore="qdrant",
    collectionname="user:project"
)
`Image Embeddings  Generation and Saving in VectorStore`

griffin_obj.generate_image_embeddings(
    new_collection=True , # OR False if collection already exists,
    imageurls=["https://samplelink/to/image1.jpg","https://samplelink/to/image2.jpg"]
)


`Text to Image Similarity Search`

search_result=griffin_obj.search_similar_images(text="query text",result_limit=1)
print(search_result)

```


> Author:Junaid Bashir
> Note:Currently Qdrant is the only supported vectorstore (new stores coming soon)
