# KV Cache Event Observer

Real-time monitoring of vLLM prefix cache events (block stored, evicted, and cache hits).

## Quick Start

```bash
# Basic monitoring (ZMQ events only)
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v

# With cache hit detection (polls Prometheus metrics)
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v -m 18081
```

## Usage

```bash
# Basic verbose monitoring
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v

# With cache hit detection (recommended for experiments)
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v -m 18081

# Run for 60 seconds
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v -d 60

# Save events to file
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v -o /tmp/events.jsonl

# Monitor worker 1 (port 20081, metrics 18082)
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20081 -v -m 18082
```

## Options

| Flag | Description |
|------|-------------|
| `-p`, `--port` | KV event ZMQ port (default: 20080, worker 1 = 20081, etc.) |
| `-m`, `--metrics-port` | Prometheus metrics port for cache hit detection (e.g., 18081) |
| `-v`, `--verbose` | Print each event as it happens |
| `-d`, `--duration` | Run for N seconds then stop |
| `-o`, `--output` | Save events to JSONL file |
| `-H`, `--host` | Worker host (default: localhost) |

## Event Types

| Symbol | Event | Source | Description |
|--------|-------|--------|-------------|
| 📦 | STORED | ZMQ | Block committed to prefix cache |
| 🗑️ | REMOVED | ZMQ | Block evicted from cache |
| 🧹 | CLEARED | ZMQ | Entire cache cleared |
| ✅ | CACHE HIT | Metrics | Tokens served from cache (requires `-m`) |

## Example Output

```
[KV Observer] Listening for KV events (msgpack multipart)...
[KV Observer] Cache hits will show as ✅ [CACHE HIT]
------------------------------------------------------------
📦 [STORED  ] seq=    32 hash=df6f76832e34d5f5 tokens= 64 medium=GPU
🗑️  [REMOVED ] seq=    33 hash=eaacc201f3aaf753 medium=GPU
✅ [CACHE HIT] tokens=  64 queried= 128 hit_rate=50%
📦 [STORED  ] seq=    34 hash=df6f76832e34d5f5 tokens= 64 medium=GPU
------------------------------------------------------------
[KV Observer] Final Statistics:
  stored_blocks: 2
  evicted_blocks: 1
  net_blocks: 1
  cache_hit_tokens: 64
  cache_query_tokens: 192
  cache_hit_rate: 33.3%
```

## Notes

- **STORED/REMOVED events**: Published via ZMQ when cache state changes
- **CACHE HIT events**: Detected by polling Prometheus metrics (requires `-m` flag)
- **No event = cache hit**: If a repeated query shows no STORED event, the block was already cached
- Events only fire for **full blocks** (64 tokens with default block size)
- Short prompts (less than 64 tokens) may not generate STORED events for incomplete blocks
- With limited cache (e.g., 16 blocks), expect frequent evictions
- **Clearing the cache**: vLLM does not expose a direct cache clear API. To fully clear the cache, restart the vLLM worker. Alternatively, use `--flush` with the experiment script to fill the cache with unique queries, pushing out old entries via LRU eviction.

## Port Mapping

| Worker | ZMQ Port (`-p`) | Metrics Port (`-m`) |
|--------|-----------------|---------------------|
| Worker 0 | 20080 | 18081 |
| Worker 1 | 20081 | 18082 |
| Worker 2 | 20082 | 18083 |

## Manual Cache Lifecycle Experiment

This experiment demonstrates the full KV cache lifecycle: **STORE → STORE → EVICT → STORE → CACHE HIT**.

### Setup

```bash
# 1. Stop any running Dynamo stack
bash stop_dynamo.sh

# 2. Configure limited cache for experiment (5 blocks)
export DYNAMO_GPU_DEVICES=0,1,2,3
export DYNAMO_TP_SIZE=4
export DYNAMO_KV_BLOCK_SIZE=64
export DYNAMO_NUM_GPU_BLOCKS_OVERRIDE=5

# 3. Start vLLM with KV events enabled
bash start_dynamo_optimized_thompson_hints_vllm.sh > startup_output.txt

# 4. In a separate terminal, start the observer
docker exec -it dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v -m 18081
```

### Run Queries

Queries must be **65+ tokens** (including chat template) to generate cache events:

```bash
# Query A (70 tokens) - STORE
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b","messages":[{"role":"user","content":"Query A: The quick brown fox jumps over the lazy dog repeatedly. The quick brown fox jumps over the lazy dog repeatedly. The quick brown fox jumps over the lazy dog."}],"max_tokens":5}'

# Query B (72 tokens) - STORE
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b","messages":[{"role":"user","content":"Query B: Pack my box with five dozen liquor jugs today please. Pack my box with five dozen liquor jugs today please. Pack my box with five dozen liquor jugs."}],"max_tokens":5}'

# Query C (76 tokens) - EVICT A, STORE C
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b","messages":[{"role":"user","content":"Query C: How vexingly quick daft zebras jump over the moon tonight. How vexingly quick daft zebras jump over the moon tonight. How vexingly quick daft zebras jump."}],"max_tokens":5}'

# Query C again - CACHE HIT
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b","messages":[{"role":"user","content":"Query C: How vexingly quick daft zebras jump over the moon tonight. How vexingly quick daft zebras jump over the moon tonight. How vexingly quick daft zebras jump."}],"max_tokens":5}'
```

### Expected Observer Output

```
📦 [STORED  ] seq=     0 hash=ca596e30d283c6f7 tokens= 64 medium=GPU   ← Query A
📦 [STORED  ] seq=     1 hash=41ccc959d03a1d21 tokens= 64 medium=GPU   ← Query B
🗑️  [REMOVED ] seq=     2 hash=ca596e30d283c6f7 medium=GPU             ← Query A evicted (LRU)
📦 [STORED  ] seq=     2 hash=b5291e07de5d51cc tokens= 64 medium=GPU   ← Query C
✅ [CACHE HIT] tokens=  64 queried=  76 hit_rate=84%                   ← Query C repeated
```

### Cache Size Guidelines

| Blocks | Usable | Behavior |
|--------|--------|----------|
| 3-4 | ~0-1 | Constant thrashing, no cache benefit |
| 5-8 | ~2-4 | Good for demonstrating evictions + hits |
| 16+ | ~10+ | Production-like behavior |

### Key Requirements

- **Prompt length**: Must exceed 64 tokens (1 block) to generate STORED events
- **Cache size**: Use `DYNAMO_NUM_GPU_BLOCKS_OVERRIDE=5` to force evictions
- **Metrics flag**: Use `-m 18081` to detect cache hits (not published via ZMQ)

## Cache Experiment Script

Run a complete A → B → C → A cache experiment:

```bash
# Basic experiment
./cache_experiment.sh

# Flush cache first (recommended)
./cache_experiment.sh --flush

# Verbose output (full API responses)
./cache_experiment.sh --flush --verbose
```

The script:
1. Optionally flushes the cache by filling it with unique queries
2. Starts the KV event observer in the background
3. Sends Query A (should STORE)
4. Sends Query B (should STORE)
5. Sends Query C (should STORE)
6. Sends Query A again (should show CACHE HIT)
7. Displays observer output and final statistics

## Requirements

- vLLM must be started with `--kv-events-config` containing `enable_kv_cache_events: true`
- The startup script `start_dynamo_optimized_thompson_hints_vllm.sh` configures this automatically when `DYNAMO_ENABLE_KV_EVENTS=true`

