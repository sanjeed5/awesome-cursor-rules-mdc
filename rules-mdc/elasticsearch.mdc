---
description: This rule file guides developers on best practices for using Elasticsearch, focusing on data modeling, query optimization, performance, and code structure for robust, high-performance search applications.
globs: **/*
---
# elasticsearch Best Practices

Elasticsearch is a powerful search engine. Treat it as such. These guidelines ensure your team builds performant, scalable, and maintainable applications.

## 1. Data Modeling

### 1.1. Define Explicit Mappings

Always define explicit mappings. This ensures data consistency, optimal indexing, and predictable search behavior. Avoid dynamic mapping for critical fields.

❌ **BAD: Relying on dynamic mapping**
```json
// Elasticsearch infers types, which can lead to 'text' for IDs or unexpected analyzers.
PUT /my_index/_doc/1
{
  "product_id": "P12345",
  "name": "Super Widget",
  "tags": ["electronics", "gadget"]
}
```

✅ **GOOD: Explicitly define field types**
```json
PUT /my_index
{
  "mappings": {
    "properties": {
      "product_id": { "type": "keyword" }, // Exact match, no analysis
      "name": { "type": "text", "analyzer": "standard" }, // Analyzed for full-text search
      "tags": { "type": "keyword" }, // Exact match for filtering/faceting
      "description": { "type": "text", "analyzer": "english" }, // Language-specific analysis
      "price": { "type": "float" },
      "created_at": { "type": "date" }
    }
  }
}
```

### 1.2. Choose Correct Field Types

Use `keyword` for exact values (IDs, tags, enums) and `text` for analyzed, full-text content.

❌ **BAD: Using `text` for exact matching or IDs**
```json
// 'product_id' as text means it will be analyzed, making exact matches inefficient.
GET /my_index/_search
{
  "query": { "match": { "product_id": "P12345" } }
}
```

✅ **GOOD: Use `keyword` for exact values**
```json
GET /my_index/_search
{
  "query": { "term": { "product_id": "P12345" } } // Efficient exact match
}
```

### 1.3. Select Appropriate Analyzers

Use lightweight, specific analyzers. `standard` is a good default. For language-specific content, use analyzers like `english`. Define custom analyzers when needed.

❌ **BAD: Using `standard` for all text fields, including highly specific ones**
```json
// 'standard' might not be optimal for highly technical or language-specific content.
{ "content": { "type": "text", "analyzer": "standard" } }
```

✅ **GOOD: Use specific or custom analyzers**
```json
PUT /my_index
{
  "settings": {
    "analysis": {
      "analyzer": {
        "code_analyzer": {
          "type": "custom",
          "tokenizer": "whitespace",
          "filter": ["lowercase"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "title": { "type": "text", "analyzer": "standard" },
      "description": { "type": "text", "analyzer": "english" },
      "code_snippet": { "type": "text", "analyzer": "code_analyzer" }
    }
  }
}
```

### 1.4. Avoid Large Documents

Keep documents well under 100MB. Large documents stress network, memory, and disk. Reconsider your unit of information (e.g., chapters instead of whole books).

❌ **BAD: Indexing an entire book as one document**
```json
// Document size can easily exceed 100MB, leading to indexing failures and poor search.
{ "book_title": "War and Peace", "full_text_content": "..." }
```

✅ **GOOD: Break down large content into smaller, searchable units**
```json
// Index chapters or paragraphs, linking back to the parent book.
{ "book_id": "B123", "chapter_title": "Chapter 1", "chapter_content": "..." }
```

### 1.5. Shard Sizing

Aim for shard sizes between 10GB and 50GB, and under 200 million documents per shard. Oversharding or huge shards degrade performance.

❌ **BAD: Default shard count for small indices, leading to many tiny shards**
```json
// An index with 1GB data and 5 primary shards is oversharded.
PUT /my_small_index { "settings": { "number_of_shards": 5 } }
```

✅ **GOOD: Calculate shards based on expected data volume and shard size goals**
```json
// For 50GB of data, 1-5 primary shards is appropriate.
PUT /my_index { "settings": { "number_of_shards": 1 } }
```

### 1.6. Use Data Streams and ILM for Time-Series

For time-series data (logs, metrics), always use data streams with Index Lifecycle Management (ILM) for automatic rollover and shard management.

✅ **GOOD: Manage time-series data with data streams and ILM**
```json
// Define an ILM policy and a component template for your data stream.
// Data streams automatically roll over backing indices based on size/age,
// preventing large, unmanageable shards.
PUT /_index_template/my_logs_template
{
  "index_patterns": ["my-logs-*"],
  "data_stream": {},
  "template": {
    "settings": { "index.lifecycle.name": "my_log_policy" },
    "mappings": { "properties": { "timestamp": { "type": "date" } } }
  }
}
```

## 2. Query Optimization

### 2.1. Avoid Returning Large Result Sets

Elasticsearch is a search engine, not a database for full table scans. For deep pagination, use the `Scroll` API. For user-facing pagination, keep `size` small (e.g., 10-100).

❌ **BAD: Attempting to retrieve all documents with `size: 10000` or more**
```json
GET /my_index/_search
{ "query": { "match_all": {} }, "size": 10000 } // High memory/network pressure
```

✅ **GOOD: Use `Scroll` API for deep pagination or bulk processing**
```json
// Initial scroll request
GET /my_index/_search?scroll=1m
{ "query": { "match_all": {} }, "size": 1000 }

// Subsequent scroll requests
GET /_search/scroll
{ "scroll": "1m", "scroll_id": "FGluY2x1ZGVfY29udGV4dF91dWlk..." }
```

### 2.2. Prefer `filter` for Non-Scoring Queries

Use `filter` clauses within a `bool` query for conditions that should not affect the relevance score (e.g., exact matches, ranges). This is more performant as filters are cached.

❌ **BAD: Using `must` for non-scoring criteria**
```json
GET /my_index/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "description": "search term" } },
        { "term": { "category": "electronics" } } // This should be a filter
      ]
    }
  }
}
```

✅ **GOOD: Use `filter` for non-scoring criteria**
```json
GET /my_index/_search
{
  "query": {
    "bool": {
      "must": { "match": { "description": "search term" } },
      "filter": { "term": { "category": "electronics" } } // Cached, no score impact
    }
  }
}
```

### 2.3. Boost Fields Judiciously

Use boosting to influence relevance scores, but test extensively. Over-boosting can lead to irrelevant results.

✅ **GOOD: Boost important fields for relevance**
```json
GET /my_index/_search
{
  "query": {
    "multi_match": {
      "query": "Elasticsearch performance",
      "fields": ["title^3", "description^1.5", "tags"] // Title is most important
    }
  }
}
```

## 3. Code Organization and Structure

### 3.1. Centralize Client Configuration

Keep Elasticsearch client settings (hosts, timeouts, retries) in a shared module or configuration file.

❌ **BAD: Hardcoding client settings in multiple places**
```python
# client.py
es_client = Elasticsearch(
    hosts=['localhost:9200'],
    timeout=30,
    max_retries=10
)
# another_module.py
another_es_client = Elasticsearch(
    hosts=['localhost:9200'],
    timeout=30,
    max_retries=10
)
```

✅ **GOOD: Centralize client configuration**
```python
# config.py
ELASTICSEARCH_HOSTS = ['https://es-cluster.mycompany.com:9200']
ELASTICSEARCH_TIMEOUT = 60
ELASTICSEARCH_MAX_RETRIES = 5
ELASTICSEARCH_API_KEY = "..." # Use API keys or username/password

# es_client.py
from elasticsearch import Elasticsearch
from .config import ELASTICSEARCH_HOSTS, ELASTICSEARCH_TIMEOUT, ELASTICSEARCH_MAX_RETRIES, ELASTICSEARCH_API_KEY

es = Elasticsearch(
    hosts=ELASTICSEARCH_HOSTS,
    timeout=ELASTICSEARCH_TIMEOUT,
    max_retries=ELASTICSEARCH_MAX_RETRIES,
    api_key=ELASTICSEARCH_API_KEY
)

# In your application code
from .es_client import es
results = es.search(index="my_index", query={"match_all": {}})
```

### 3.2. Centralize Mapping Definitions

Store index mapping definitions in version-controlled JSON/YAML files. Apply them programmatically.

❌ **BAD: Defining mappings ad-hoc or inline in application code**
```python
# In some script
es.indices.create(index="products", body={"mappings": {"properties": {"name": {"type": "text"}}}})
```

✅ **GOOD: Store mappings in dedicated files**
```json
// mappings/products.json
{
  "properties": {
    "product_id": { "type": "keyword" },
    "name": { "type": "text", "analyzer": "standard" },
    "description": { "type": "text", "analyzer": "english" },
    "price": { "type": "float" },
    "available": { "type": "boolean" }
  }
}
```
```python
# scripts/init_indices.py
import json
from es_client import es

def create_index(index_name, mapping_file):
    with open(f"mappings/{mapping_file}.json", "r") as f:
        mapping = json.load(f)
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, mappings=mapping)
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")

create_index("products", "products")
```

## 4. Performance Considerations

### 4.1. Bulk Indexing

Always use the bulk API for indexing multiple documents. This significantly reduces network overhead and improves indexing throughput.

❌ **BAD: Indexing documents one by one in a loop**
```python
for doc in documents:
    es.index(index="my_index", id=doc["id"], document=doc)
```

✅ **GOOD: Use the bulk API**
```python
from elasticsearch.helpers import bulk

actions = [
    {"_index": "my_index", "_id": doc["id"], "_source": doc}
    for doc in documents
]
bulk(es, actions)
```

### 4.2. Monitor and Benchmark

Continuously monitor your cluster's health, performance, and resource usage. Benchmark with realistic data and query loads before deploying changes.

✅ **GOOD: Use Kibana's monitoring tools and run load tests**
```bash
# Example: Use a tool like k6 or JMeter to simulate production load
k6 run load_test_script.js --vus 100 --duration 30s
```

## 5. Common Pitfalls

### 5.1. `fielddata` on `text` fields

Enabling `fielddata` on `text` fields for sorting or aggregations consumes large amounts of heap memory and can lead to `OutOfMemoryError`. Use `keyword` fields for these operations.

❌ **BAD: Enabling `fielddata` on a `text` field**
```json
PUT /my_index/_mapping
{
  "properties": {
    "description": { "type": "text", "fielddata": true } // Avoid this!
  }
}
```

✅ **GOOD: Use `keyword` fields for sorting/aggregations**
```json
PUT /my_index/_mapping
{
  "properties": {
    "description": {
      "type": "text",
      "fields": {
        "keyword": { "type": "keyword", "ignore_above": 256 } // Use this for sorting/aggregations
      }
    }
  }
}
```

### 5.2. Oversharding

Too many small shards degrade search performance and cluster stability. Each shard has overhead.

❌ **BAD: Creating an index with 100 shards for 10GB of data**
```json
PUT /my_index { "settings": { "number_of_shards": 100 } }
```

✅ **GOOD: Target 10-50GB per shard**
```json
// For 10GB data, 1 shard is sufficient.
PUT /my_index { "settings": { "number_of_shards": 1 } }
```

## 6. Security Best Practices

### 6.1. Enable Security Features

Always enable Elasticsearch security (authentication, authorization, TLS/SSL) in production. Use API keys or role-based access control.

❌ **BAD: Running Elasticsearch without security in production**
```bash
# No authentication, anyone can access
```

✅ **GOOD: Configure API keys and TLS**
```python
from elasticsearch import Elasticsearch

es = Elasticsearch(
    hosts=['https://es-cluster.mycompany.com:9200'],
    api_key=("id", "api_key_value"), # Use API keys
    verify_certs=True,
    ca_certs="/path/to/ca.crt"
)
```

### 6.2. Network Isolation

Restrict network access to Elasticsearch nodes. Only allow necessary traffic from application servers.

✅ **GOOD: Use firewalls/security groups to limit access**
```bash
# Example: AWS Security Group rule
# Inbound: TCP 9200 from application server IPs only
```