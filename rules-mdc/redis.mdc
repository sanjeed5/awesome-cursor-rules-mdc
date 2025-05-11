---
description: This rule provides best practices for working with Redis, covering code organization, performance, security, testing, and common pitfalls to ensure efficient and reliable usage. It applies to any language file interacting with Redis.
globs: *.py,*.js,*.go,*.java,*.c,*.cpp,*.rb,*.php,*.ts,*.rs,*.kt,*.scala
---
- **General Best Practices**
  - Follow official Redis documentation and community best practices.
  - Use the most recent stable version of the Redis client library for your language.
  - Regularly update the Redis server and client libraries to benefit from bug fixes, performance improvements, and security patches.

- **Connection Management**
  - **Connection Pooling:** Use connection pooling to reduce the overhead of creating new connections for each operation.  This significantly improves performance, especially in high-traffic scenarios.
  - **Connection Timeout:** Configure appropriate connection timeouts to prevent indefinite blocking in case of network issues.
  - **Retry Logic:** Implement retry logic with exponential backoff for transient connection errors.
  - **Secure Connections:**  When connecting to a remote Redis instance, always use TLS/SSL encryption (redis+ssl://) to protect data in transit, especially when dealing with sensitive information. Ensure proper certificate validation.

- **Data Modeling and Serialization**
  - **Key Naming:** Use consistent and meaningful key naming conventions. Consider using namespaces or prefixes to organize keys and avoid collisions.
  - **Data Serialization:** Choose an efficient serialization format (e.g., JSON, Protocol Buffers, MessagePack) and use it consistently.  Consider the trade-offs between human readability, storage space, and serialization/deserialization performance.
  - **Smaller Values:**  Redis works best with smaller values. Break larger data structures into smaller chunks and distribute them across multiple keys.  This improves memory utilization and performance.
  - **Data Types:**  Leverage Redis's rich data types (strings, lists, sets, sorted sets, hashes, streams) effectively to optimize data storage and operations.  Choose the data type that best suits the use case.

- **Command Usage**
  - **Transactions:** Use transactions (`MULTI`, `EXEC`, `DISCARD`, `WATCH`) to ensure atomicity when performing multiple operations.  Be aware of the limitations of Redis transactions (no rollback on individual command failure).
  - **Pipelines:** Use pipelines to batch multiple commands together and reduce network round-trip time. This dramatically improves performance for bulk operations.
  - **Lua Scripting:** Use Lua scripting for complex operations that require atomicity and server-side processing.  This reduces network traffic and improves performance compared to executing multiple individual commands.
  - **Avoid Blocking Commands:**  Avoid using blocking commands like `KEYS`, `FLUSHALL`, `FLUSHDB`, `SORT` without `LIMIT` in production environments. These commands can block the Redis server and degrade performance.
  - **Use SCAN:** Instead of `KEYS`, use the `SCAN` command for iterating over keys in a non-blocking manner.  This allows the Redis server to continue serving other requests while iterating.
  - **Efficient Deletion:** Instead of `FLUSHALL` or `FLUSHDB`, use `SCAN` with `DEL` to delete keys in batches, minimizing disruption to the server.
  - **TTL Management:** Set appropriate Time-To-Live (TTL) values for keys to automatically expire data that is no longer needed. This helps manage memory usage and prevent data from becoming stale.

- **Memory Management**
  - **Maxmemory:** Configure the `maxmemory` directive to limit the amount of memory Redis can use.  When the limit is reached, Redis will evict keys based on the configured eviction policy.
  - **Eviction Policies:** Choose an appropriate eviction policy (e.g., `LRU`, `LFU`, `volatile-ttl`) based on your application's needs. Understand the trade-offs between different eviction policies.
  - **Memory Fragmentation:** Monitor memory fragmentation and consider restarting Redis periodically to defragment memory. The `INFO memory` command provides information about memory usage and fragmentation.

- **Performance Monitoring and Tuning**
  - **Redis Monitor:** Use the `MONITOR` command (with caution in production) to observe real-time commands being executed on the server.  This can help identify performance bottlenecks.
  - **Redis Slow Log:** Configure the Redis slow log to record commands that take longer than a specified amount of time to execute. Analyze the slow log to identify performance issues.
  - **INFO Command:** Use the `INFO` command to gather information about the Redis server, including memory usage, CPU usage, and client connections.  This information can be used to monitor performance and identify potential problems.
  - **Latency Monitoring:** Monitor Redis latency using tools like `redis-cli --latency` or dedicated monitoring solutions.  High latency can indicate performance issues.

- **Security Considerations**
  - **Authentication:**  Enable authentication using the `requirepass` directive to protect the Redis server from unauthorized access.  Use a strong password.
  - **Access Control Lists (ACLs):** Use ACLs to restrict access to specific commands and keys for different users.  This provides fine-grained control over access to Redis data.
  - **Network Security:**  Restrict network access to the Redis server using firewalls or other network security measures.  Only allow connections from trusted sources.
  - **Disable Unsafe Commands:**  Disable or rename potentially dangerous commands like `FLUSHALL`, `FLUSHDB`, `KEYS`, `EVAL` using the `rename-command` directive.  This reduces the risk of accidental or malicious misuse.
  - **Regular Audits:** Conduct regular security audits of the Redis configuration and usage patterns to identify and address potential vulnerabilities.
  - **Input Validation:** Always validate and sanitize any data being stored in Redis to prevent injection attacks.

- **Testing Strategies**
  - **Unit Tests:** Write unit tests to verify the functionality of your code that interacts with Redis.  Use mocking or stubbing to isolate the Redis interactions from the rest of the code.
  - **Integration Tests:** Write integration tests to verify the interaction between your code and the Redis server.  Use a dedicated test Redis instance for integration tests.
  - **End-to-End Tests:** Write end-to-end tests to verify the entire application flow, including the interaction with Redis.  This ensures that the application works correctly in a realistic environment.
  - **Data Population:** When testing, consider populating your redis database with a representative set of data. 
  - **Test Organization:** Organize tests logically, separating unit, integration, and end-to-end tests into different directories or modules.
  - **Mocking and Stubbing:** Use mocking and stubbing frameworks to simulate Redis behavior during unit tests.

- **Code Organization and Structure**
  - **Dedicated Module:** Create a dedicated module or class to encapsulate all Redis-related operations.  This promotes code reuse and maintainability.
  - **Configuration Management:** Store Redis connection parameters (host, port, password) in a configuration file or environment variables.  This makes it easy to change the Redis configuration without modifying code.
  - **Abstraction Layer:**  Consider creating an abstraction layer on top of the Redis client library to provide a higher-level API for your application.  This can improve code readability and make it easier to switch to a different Redis client library in the future.

- **Common Pitfalls and Gotchas**
  - **N+1 Problem:** Avoid the N+1 problem when retrieving data from Redis.  Instead of making multiple individual requests, use pipelining or Lua scripting to retrieve the data in a single request.
  - **Race Conditions:** Be aware of potential race conditions when updating data in Redis.  Use transactions or Lua scripting to ensure atomicity.
  - **Large Values:** Avoid storing extremely large values in Redis.  This can lead to performance issues and memory exhaustion.
  - **Key Expiration:**  Be careful when using key expiration (TTL).  If keys expire unexpectedly, it can lead to data loss or inconsistent application behavior.
  - **Blocking Operations in Event Loops:** Do not perform blocking Redis operations directly in event loops or GUI threads. Use asynchronous operations instead to avoid blocking the main thread.

- **Tooling and Environment**
  - **Redis CLI:** Use the Redis CLI (`redis-cli`) for interacting with the Redis server, executing commands, and monitoring performance.
  - **Redis Desktop Manager:** Use a Redis desktop manager (e.g., RedisInsight, Medis) for visualizing data, managing keys, and monitoring the Redis server.
  - **Linting and Formatting:** Configure linters and formatters to enforce consistent code style and best practices in your Redis-related code.
  - **CI/CD Integration:** Integrate Redis testing and deployment into your CI/CD pipeline to automate the testing and deployment process.
  - **Monitoring Tools:** Utilize monitoring tools like Prometheus, Grafana, or Datadog to monitor Redis performance and health in production environments.