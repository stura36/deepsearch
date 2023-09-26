# Project overview
The Deepsearch project is a simple-to-use desktop application that provides semantic image search through the desktop computer file system.

Under the hood," Deepsearch uses a composite model comprising of two well-known models, ResNet and SBERT. 
ResNet creates semantically aware vectors for images, which are fed into the search database. 
In the same manner for textual queries, we use SBERT. 
Both the image and query vectors produced by each respective model are then linearly transformed into a shared vector space, 
where cosine similarity between points encapsulates semantic similarity between text and image vectors. 
Using cosine similarity, Deepsearch retrieves the closest image for the given textual query.




