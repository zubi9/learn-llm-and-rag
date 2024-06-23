# Lecture 1: Introduction to Retrieval Augmented Generation ChatBot

Welcome to Lecture 1 of our course on building Retrieval Augmented Generation (RAG) ChatBots. In this lecture, we focus on two document search techniques after converting Google Docx files' data into a JSON format. These techniques are used to retrieve relevant documents and enhance the context of the questions for better answer generation using the OpenAI API.

## Lecture Outline

### Overview
- Introduction to RAG and its importance in building intelligent ChatBots.
- Overview of the two document search techniques: Minisearch and Elasticsearch.

### Document Conversion
- Fetching Google Docx files and converting their data into a JSON format.

### Search Techniques
1. **Minisearch:**
   - A mini search engine-like program that retrieves documents based on the TF-IDF similarity algorithm.
   - Steps to implement Minisearch and how it works.
   
2. **Elasticsearch:**
   - A widely used search algorithm in e-commerce websites for searching the most relevant keywords.
   - Steps to set up and use Elasticsearch for document retrieval.

### Enhancing Query Context
- Adding retrieved text to the original query to enhance the context for better answer generation.
- Using the OpenAI API key for generating answers with the enhanced query.

## Getting Started

### Prerequisites
- Python >=3.10
- OpenAI API key
- Elasticsearch (for Elasticsearch technique)

# Module 1: Introduction
 
In this module, we will learn what LLM and RAG are and
implement a simple RAG pipeline to answer questions about 
the FAQ Documents from our Zoomcamp courses

What we will do: 

* Index Zoomcamp FAQ documents
    * DE Zoomcamp: https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit
    * ML Zoomcamp: https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit
    * MLOps Zoomcamp: https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit
* Create a Q&A system for answering questions about these documents 

## 1.1 Introduction to LLM and RAG

<a href="https://www.youtube.com/watch?v=Q75JgLEXMsM&list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R">
  <img src="https://markdown-videos-api.jorgenkh.no/youtube/Q75JgLEXMsM">
</a>

* LLM
* RAG
* RAG architecture
* Course outcome


## 1.2 Preparing the Environment

<a href="https://www.youtube.com/watch?v=ozCpmkbJNJE&list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R">
  <img src="https://markdown-videos-api.jorgenkh.no/youtube/ozCpmkbJNJE">
</a>

* Installing libraries
* Alternative: installing anaconda or miniconda

```bash
pip install tqdm notebook==7.1.2 openai elasticsearch pandas scikit-learn
```

## 1.3 Retrieval

<a href="https://www.youtube.com/watch?v=olvem333Bqo&list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R">
  <img src="https://markdown-videos-api.jorgenkh.no/youtube/olvem333Bqo">
</a>

* We will use the search engine we build in the [build-your-own-search-engine workshop](https://github.com/alexeygrigorev/build-your-own-search-engine): [minsearch](https://github.com/alexeygrigorev/minsearch)
* Indexing the documents
* Peforming the search


## 1.4 Generation with OpenAI

<a href="https://www.youtube.com/watch?v=qz316T3U49Q&list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R">
  <img src="https://markdown-videos-api.jorgenkh.no/youtube/qz316T3U49Q">
</a>

* Invoking OpenAI API
* Building the prompt
* Getting the answer


If you don't want to use a service, you can run an LLM locally
refer to [module 2](../02-open-source/) for more details.

In particular, check "2.7 Ollama - Running LLMs on a CPU" - 
it can work with OpenAI API, so to make the example from 1.4 
work locally, you only need to change a few lines of code.


## 1.4.2 OpenAI API Alternatives

<a href="https://www.youtube.com/watch?v=HObjFso2UJE&list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R">
  <img src="https://markdown-videos-api.jorgenkh.no/youtube/HObjFso2UJE">
</a>

[Open AI Alternatives](open-ai-alternatives.md)


## 1.5 Cleaned RAG flow

<a href="https://www.youtube.com/watch?v=vkTiVwwch6A&list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R">
  <img src="https://markdown-videos-api.jorgenkh.no/youtube/vkTiVwwch6A">
</a>

* Cleaning the code we wrote so far
* Making it modular

## 1.6 Searching with ElasticSearch

<a href="https://www.youtube.com/watch?v=1lgbR5wMvsI&list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R">
  <img src="https://markdown-videos-api.jorgenkh.no/youtube/1lgbR5wMvsI">
</a>

* Run ElasticSearch with Docker
* Index the documents
* Replace MinSearch with ElasticSearch

Running ElasticSearch:

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

Index settings:

```python
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
```

Query:

```python
{
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```

We use `"type": "best_fields"`. You can read more about 
different types of `multi_match` search in [elastic-search.md](elastic-search.md).

# 1.7 Homework
More information [here](../cohorts/2024/01-intro/homework.md).

# Notes

* [Notes by slavaheroes](https://github.com/slavaheroes/llm-zoomcamp/blob/homeworks/01-intro/notes.md)
* Did you take notes? Add them above this line (Send a PR with *links* to your notes)

## Conclusion

In this lecture, you learned how to build a basic RAG ChatBot using two different document search techniques: Minisearch and Elasticsearch. You also learned how to enhance query context and generate better answers using the OpenAI API. 

Feel free to explore the scripts and modify them to suit your needs. Happy learning!