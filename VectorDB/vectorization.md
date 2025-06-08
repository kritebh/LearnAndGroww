## Vector DB

A vector database is a specialized database designed to store and retrieve high-dimensional vectors (e.g., 128D, 768D) and perform similarity search — especially useful for things like:

	- Semantic search
	- Image or audio search
	- Recommendation systems
	- AI assistants


Text must be translated into numbers because ML models can’t understand raw text. The conversion is done using a process called text embedding, which maps each piece of text to a vector in a high-dimensional space.

The key idea is:

	- Similar meanings → similar vectors
	- Different meanings → different vectors

### How Vector DB Works

1. Text Embedding

First, you convert text to vectors using models like BERT, OpenAI, Cohere, etc.

2. Indexing

The vectors are stored in a special data structure for fast approximate nearest neighbor (ANN) search. Common techniques include:
	•	FAISS (Facebook AI)
	•	HNSW (Hierarchical Navigable Small World)
	•	IVF, PQ, etc.

3. Similarity Search

When a query vector is provided, the DB finds the most similar vectors (nearest neighbors) using metrics like:
	•	Cosine similarity
	•	Euclidean distance
	•	Dot product

#### Popular Vector Databases

| Name     | Highlights                                           |
|----------|------------------------------------------------------|
| Pinecone | Fully managed, easy to scale                         |
| Weaviate | Open source, supports hybrid (text + vector) search  |
| Qdrant   | Rust-based, fast, good for large-scale               |
| Milvus   | High-performance, used at enterprise scale           |
| FAISS    | Not a full DB, but a powerful vector index library from Meta |


## Vectorization Article Reference

[Pinecone Vector DB](https://www.pinecone.io/learn/vector-database/)

[Indexing For Vector](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/)

[Vector Embedding](https://www.pinecone.io/learn/vector-embeddings-for-developers/)

[Getting Started With Embeddings HF](https://huggingface.co/blog/getting-started-with-embeddings)

## Vectorization from scikit

``` python 

from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love dog", "Cat loves me","I love my dog","me loves cat"]
# texts = ["king", "dog","queen","puppy","dogs"]
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(texts).toarray()

print("Vocabulary:", vectorizer.vocabulary_)
print("Vectors:")
print(vectors)


#output
Vocabulary: {'love': 2, 'dog': 1, 'cat': 0, 'loves': 3, 'me': 4, 'my': 5}
Vectors:
[[0 1 1 0 0 0]
 [1 0 0 1 1 0]
 [0 1 1 0 0 1]
 [1 0 0 1 1 0]]

```

## Popular Algorithms for Vector Indexing

| Algorithm              | Full Name                              | Key Idea                                 | Strength                           |
| ---------------------- | -------------------------------------- | ---------------------------------------- | ---------------------------------- |
| **Flat (Brute-force)** | -                                      | Compare every vector                     | 100% accurate, simple              |
| **IVF**                | Inverted File Index                    | Partitions vector space into “buckets”   | Fast with large datasets           |
| **HNSW**               | Hierarchical Navigable Small World     | Builds a multi-layer graph               | Very fast, high recall             |
| **Annoy**              | Approximate Nearest Neighbors Oh Yeah  | Builds multiple random trees             | Fast + low memory usage            |
| **PQ**                 | Product Quantization                   | Compresses vectors into smaller chunks   | Saves memory                       |
| **ScaNN**              | Scalable Nearest Neighbors (by Google) | Combines partitioning and quantization   | High accuracy + speed              |
| **LSH**                | Locality Sensitive Hashing             | Hashes similar vectors into same buckets | Good for binary search-like speed  |
| **NSG**                | Navigating Spreading-out Graph         | Optimized graph similar to HNSW          | High performance in large datasets |


