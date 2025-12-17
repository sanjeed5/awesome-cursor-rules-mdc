---
description: This guide provides definitive, actionable best practices for using Redis effectively, focusing on data modeling, performance, security, and cluster-aware client usage to build robust and scalable applications.
globs: **/*
---
# Redis Best Practices

Redis is a powerful, single-threaded, in-memory database. Its efficiency hinges on how you interact with it. Follow these rules to ensure your applications are performant, scalable, and resilient.

## 1. Data Modeling & Key Design

Efficient key design is paramount for performance and memory.

### 1.1. Use Hierarchical Key Naming

Adopt a consistent, hierarchical key naming convention. This improves readability and allows for logical grouping.

❌ BAD
```redis
user:1:name
user:1:email
product:123:details
order:456:items
```

✅ GOOD
```redis
user:1:profile:name
user:1:profile:email
product:123:data
order:456:cart:items
```
*Rationale*: Clearer separation, better organization.

### 1.2. Avoid Large Objects in Single Keys

Do not store multi-megabyte objects in a single key. Break down complex data into hashes or streams. This improves eviction, replication, and memory efficiency.

❌ BAD
```python
# Storing a large JSON object directly as a string
user_data = {"id": 1, "name": "Alice", "preferences": {...large_object...}, "history": [...many_items...]}
redis_client.set(f"user:1:data", json.dumps(user_data))
```

✅ GOOD
```python
# Using a Redis Hash for structured data
redis_client.hset(f"user:1:profile", mapping={
    "id": 1,
    "name": "Alice",
    "email": "alice@example.com"
})
# Storing large, evolving data in a stream or separate keys
redis_client.xadd(f"user:1:activity_stream", {"action": "login", "timestamp": time.time()})
```
*Rationale*: Hashes are optimized for field-value pairs, streams for time-series data. This keeps individual values small, aiding Redis's single-threaded nature.

### 1.3. Use Hashtags for Multi-Key Operations in Clusters

When performing multi-key operations (e.g., transactions, Lua scripts) in a Redis Cluster, ensure all involved keys reside on the same hash slot by using hashtags.

❌ BAD
```python
# These keys will likely be on different slots, causing MOVED errors in a cluster
redis_client.mget("user:1:profile", "user:1:orders")
```

✅ GOOD
```python
# Keys with the same hashtag ({...}) are guaranteed to be on the same slot
redis_client.mget("{user:1}:profile", "{user:1}:orders")
```
*Rationale*: Hashtags (`{...}`) force Redis to hash only the content within the braces, ensuring co-location for atomic operations.

### 1.4. Never Use Numbered Databases (`SELECT`)

Numbered databases (`SELECT`) are an anti-pattern. They provide false isolation and are not supported by Redis Cluster.

❌ BAD
```python
redis_client.select(1) # Using database 1
redis_client.set("mykey", "myvalue")
```

✅ GOOD
```python
# Use distinct key prefixes instead of numbered databases
redis_client.set("app:cache:mykey", "myvalue")
redis_client.set("app:sessions:mykey", "myvalue")
```
*Rationale*: Numbered databases share the same underlying instance, meaning operations like `KEYS` on one database block all others. Key prefixes are the correct way to logically separate data.

## 2. Connection & Memory Management

Proper client configuration and instance sizing are critical.

### 2.1. Always Use Connection Pooling

Creating new TCP connections for every Redis command is expensive. Use a connection pool to reuse connections.

❌ BAD
```python
# Creates a new connection for each operation
redis_client = redis.Redis(host='localhost', port=6379)
redis_client.set("key", "value")
redis_client.get("key")
```

✅ GOOD
```python
# Uses a connection pool (default for most clients, but explicitly shown)
pool = redis.ConnectionPool(host='localhost', port=6379, max_connections=10)
redis_client = redis.Redis(connection_pool=pool)
redis_client.set("key", "value")
redis_client.get("key")
```
*Rationale*: Connection pooling significantly reduces overhead and improves latency.

### 2.2. Monitor and Manage Memory Usage

Configure `maxmemory` policies and lazy freeing. Provision instances with sufficient headroom for write spikes.

```redis
# redis.conf
maxmemory 8gb             # Set maximum memory usage
maxmemory-policy allkeys-lru # Evict least recently used keys when memory limit is reached
lazyfree-lazy-eviction yes # Asynchronously free memory during eviction
lazyfree-lazy-expire yes   # Asynchronously free memory during key expiration
lazyfree-lazy-server-del yes # Asynchronously free memory when deleting keys
```
*Rationale*: Proactive memory management prevents OOM errors and maintains predictable performance.

## 3. Cluster-Aware Client Usage

When using Redis Cluster, your client must understand the cluster topology.

### 3.1. Use Cluster-Aware Clients

Always use client libraries specifically designed for Redis Cluster (e.g., `redis-py-cluster`, Lettuce, GLIDE). These clients handle slot mapping and `MOVED` redirections automatically.

❌ BAD
```python
# This client is not cluster-aware and will fail or perform poorly
import redis
client = redis.Redis(host='cluster-node-1', port=6379)
client.set("mykey", "myvalue")
```

✅ GOOD
```python
# Use a cluster-aware client
from redis.cluster import RedisCluster
client = RedisCluster(host='cluster-node-1', port=6379, decode_responses=True)
client.set("mykey", "myvalue")
```
*Rationale*: Cluster-aware clients prevent `MOVED` errors and ensure requests are routed to the correct node, enabling seamless scaling.

### 3.2. Implement Exponential Backoff for Client Discovery

When clients disconnect or need to discover cluster topology, implement exponential backoff with jitter on retries to prevent overwhelming the cluster with `CLUSTER SLOTS` commands.

```python
import time
import random
from redis.cluster import RedisCluster, RedisClusterException

def connect_with_backoff(hosts, max_retries=5):
    for i in range(max_retries):
        try:
            client = RedisCluster(startup_nodes=hosts, decode_responses=True)
            client.ping() # Verify connection
            return client
        except RedisClusterException as e:
            print(f"Connection failed (attempt {i+1}/{max_retries}): {e}")
            sleep_time = (2 ** i) + random.uniform(0, 1) # Exponential backoff with jitter
            time.sleep(sleep_time)
    raise ConnectionError("Failed to connect to Redis Cluster after multiple retries")

# Example usage
startup_nodes = [{"host": "node1", "port": 6379}, {"host": "node2", "port": 6379}]
redis_cluster_client = connect_with_backoff(startup_nodes)
```
*Rationale*: Prevents a thundering herd problem during cluster reconfigurations or outages.

## 4. Query Optimization & Batch Operations

Minimize network round-trips and avoid resource-intensive commands.

### 4.1. Use Pipelining for Batch Operations

Group multiple commands into a single request to reduce network latency.

❌ BAD
```python
# Multiple round-trips
redis_client.set("key1", "value1")
redis_client.set("key2", "value2")
redis_client.get("key1")
redis_client.get("key2")
```

✅ GOOD
```python
# Single round-trip for multiple commands
pipe = redis_client.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.get("key1")
pipe.get("key2")
results = pipe.execute()
```
*Rationale*: Pipelining batches commands, sending them in one go and receiving all results at once, drastically improving throughput.

### 4.2. Avoid `KEYS` in Production

`KEYS` is a blocking, O(N) operation that can halt your Redis server. Use `SCAN` for iterating keys.

❌ BAD
```python
# Blocks the server for large datasets
all_keys = redis_client.keys("user:*")
```

✅ GOOD
```python
# Iterates keys incrementally without blocking
cursor = 0
while True:
    cursor, keys = redis_client.scan(cursor, match="user:*", count=100)
    for key in keys:
        # Process key
        pass
    if cursor == 0:
        break
```
*Rationale*: `SCAN` provides an iterator-like interface, distributing the work over multiple calls and preventing server stalls.

### 4.3. Limit Result Sets for Collections

Avoid fetching entire large lists, sets, or hashes. Use `LRANGE` with limits, `HSCAN`, or `SSCAN`.

❌ BAD
```python
# Fetches all members of a potentially huge set
all_members = redis_client.smembers("my_large_set")
```

✅ GOOD
```python
# Iterates members incrementally
cursor = 0
while True:
    cursor, members = redis_client.sscan(cursor, name="my_large_set", count=100)
    for member in members:
        # Process member
        pass
    if cursor == 0:
        break
```
*Rationale*: Prevents excessive memory usage on both the server and client, and reduces network transfer.

## 5. Security Best Practices

Secure your Redis instances from unauthorized access.

### 5.1. Always Set a Strong Password

Never run a Redis instance without authentication enabled, especially in production.

❌ BAD
```redis
# No password configured, highly insecure
# redis.conf (default)
# requirepass foobared
```

✅ GOOD
```redis
# redis.conf
requirepass your_strong_and_unique_password_here
```
*Rationale*: An unprotected Redis instance is a common target for attackers.

### 5.2. Isolate Redis Instances

Deploy Redis behind a firewall, in a private network, or within a VPC. Only allow access from trusted application servers.

*Rationale*: Reduces the attack surface by preventing direct public access.

## 6. Testing Approaches

Ensure your Redis interactions are robust and performant.

### 6.1. Use a Dedicated Test Redis Instance

For unit and integration tests, spin up a clean Redis instance (e.g., via Docker) to ensure isolated and repeatable tests.

```python
# Example using pytest and a local Redis fixture
import pytest
import redis
import docker

@pytest.fixture(scope="session")
def redis_container():
    client = docker.from_env()
    container = client.containers.run('redis:latest', detach=True, ports={'6379/tcp': 6379})
    time.sleep(2) # Give Redis time to start
    yield container
    container.stop()
    container.remove()

@pytest.fixture(scope="function")
def redis_client(redis_container):
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    client.flushdb() # Clear database for each test
    yield client
    client.flushdb()
```
*Rationale*: Prevents test pollution and ensures tests run against a known state.

### 6.2. Conduct Load Testing

Simulate real-world traffic patterns to identify performance bottlenecks and validate scaling strategies. Tools like `memtier_benchmark` are ideal.

```bash
# Example memtier_benchmark command
memtier_benchmark -s localhost -p 6379 -n 100000 -c 50 -t 4 --ratio=1:1
```
*Rationale*: Uncovers issues related to connection limits, memory pressure, and command latency under stress.