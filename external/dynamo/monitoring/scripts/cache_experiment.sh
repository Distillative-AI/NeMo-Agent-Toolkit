#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# KV Cache Experiment Script
# 
# Demonstrates prefix cache behavior with queries A → B → C → A
# Shows: STORED events, REMOVED (eviction) events, and CACHE HITs
#
# Usage:
#   ./cache_experiment.sh              # Run with defaults
#   ./cache_experiment.sh --flush      # Flush cache first by filling it
#   ./cache_experiment.sh --verbose    # Show full curl responses

set -euo pipefail

# Configuration
API_URL="${DYNAMO_API_URL:-http://localhost:8000}"
MODEL="${DYNAMO_MODEL_NAME:-llama-3.3-70b}"
ZMQ_PORT="${DYNAMO_KV_EVENT_PORT:-20080}"
METRICS_PORT="${DYNAMO_WORKER_METRICS_PORT:-18081}"
MAX_TOKENS=5
VERBOSE=false
FLUSH_CACHE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v) VERBOSE=true; shift ;;
        --flush|-f) FLUSH_CACHE=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--flush]"
            echo "  --verbose, -v  Show full API responses"
            echo "  --flush, -f    Flush cache by filling it before experiment"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=========================================================${NC}"
echo -e "${CYAN}   KV Cache Experiment: A → B → C → A${NC}"
echo -e "${CYAN}=========================================================${NC}"
echo ""
echo -e "API:          ${API_URL}"
echo -e "Model:        ${MODEL}"
echo -e "ZMQ Port:     ${ZMQ_PORT}"
echo -e "Metrics Port: ${METRICS_PORT}"
echo ""

# Check if API is available
echo -e "${BLUE}Checking API availability...${NC}"
if ! curl -s --max-time 5 "${API_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: API not available at ${API_URL}${NC}"
    echo "Make sure Dynamo is running: bash start_dynamo_optimized_thompson_hints_vllm.sh"
    exit 1
fi
echo -e "${GREEN}✓ API is available${NC}"
echo ""

# Long prompts that will fill at least 1 complete block (64 tokens each)
# Each prompt is ~120+ tokens to ensure at least 1 full block is stored
QUERY_A="Query Alpha: Please provide a comprehensive and detailed explanation of quantum computing technology. Start by explaining what quantum bits (qubits) are and how they fundamentally differ from classical binary bits. Then thoroughly discuss the principle of quantum superposition and how it enables massive parallelism in quantum computations."

QUERY_B="Query Beta: Please provide an in-depth explanation of machine learning and artificial intelligence. Begin by describing the fundamental differences between supervised, unsupervised, and reinforcement learning paradigms. Then explain neural network architectures including feedforward networks, convolutional neural networks, and transformers."

QUERY_C="Query Charlie: Please provide a detailed overview of cloud computing infrastructure and services. Start by explaining the differences between Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). Then discuss containerization technologies like Docker and Kubernetes orchestration."

# Function to send a query and display results
send_query() {
    local name=$1
    local prompt=$2
    local color=$3
    
    echo -e "${color}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${color}Sending Query ${name}${NC}"
    echo -e "${color}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    response=$(curl -s "${API_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
            \"max_tokens\": ${MAX_TOKENS}
        }")
    
    if [ "$VERBOSE" = true ]; then
        echo "$response" | jq .
    else
        prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // "N/A"')
        completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // "N/A"')
        echo -e "  Prompt tokens:     ${prompt_tokens}"
        echo -e "  Completion tokens: ${completion_tokens}"
    fi
    echo ""
}

# Function to flush cache by sending many unique queries
flush_cache() {
    echo -e "${YELLOW}Flushing cache by filling it with unique queries...${NC}"
    echo -e "${YELLOW}(This may take a minute)${NC}"
    echo ""
    
    for i in $(seq 1 20); do
        curl -s "${API_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${MODEL}\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Flush query number ${i}: This is a unique cache flush query designed to evict existing cached blocks from the prefix cache. Random identifier: ${RANDOM}${RANDOM}${RANDOM}. Please provide a detailed explanation of topic ${i}.\"}],
                \"max_tokens\": 1
            }" > /dev/null 2>&1
        echo -ne "\r  Progress: ${i}/20"
    done
    echo -e "\n${GREEN}✓ Cache flushed${NC}"
    echo ""
}

# Get initial cache metrics
echo -e "${BLUE}Initial cache state:${NC}"
initial_hits=$(curl -s "http://localhost:${METRICS_PORT}/metrics" | grep "vllm:prefix_cache_hits_total{" | grep -oE '[0-9.]+$' || echo "0")
initial_queries=$(curl -s "http://localhost:${METRICS_PORT}/metrics" | grep "vllm:prefix_cache_queries_total{" | grep -oE '[0-9.]+$' || echo "0")
echo -e "  Cache hits:    ${initial_hits}"
echo -e "  Cache queries: ${initial_queries}"
echo ""

# Flush cache if requested
if [ "$FLUSH_CACHE" = true ]; then
    flush_cache
fi

# Start the KV event observer in the background
echo -e "${BLUE}Starting KV event observer...${NC}"
OBSERVER_LOG=$(mktemp)
docker exec dynamo-vllm python /workspace/monitoring/scripts/kv_event_observer.py \
    -p "${ZMQ_PORT}" -v -m "${METRICS_PORT}" -d 60 > "$OBSERVER_LOG" 2>&1 &
OBSERVER_PID=$!
sleep 2
echo -e "${GREEN}✓ Observer started (PID: ${OBSERVER_PID})${NC}"
echo ""

echo -e "${CYAN}=========================================================${NC}"
echo -e "${CYAN}   Starting Query Sequence: A → B → C → A${NC}"
echo -e "${CYAN}=========================================================${NC}"
echo ""

# Send queries with delays to allow event processing
send_query "A (first time)" "$QUERY_A" "$GREEN"
sleep 2

send_query "B" "$QUERY_B" "$YELLOW"
sleep 2

send_query "C" "$QUERY_C" "$RED"
sleep 2

send_query "A (repeated - expect cache hit)" "$QUERY_A" "$GREEN"
sleep 3

# Stop observer and show results
echo -e "${CYAN}=========================================================${NC}"
echo -e "${CYAN}   Stopping Observer & Showing Results${NC}"
echo -e "${CYAN}=========================================================${NC}"
echo ""

# Kill observer gracefully
kill $OBSERVER_PID 2>/dev/null || true
wait $OBSERVER_PID 2>/dev/null || true
sleep 1

# Display observer output
echo -e "${BLUE}KV Event Observer Output:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
cat "$OBSERVER_LOG"
echo ""

# Get final cache metrics
echo -e "${BLUE}Final cache state:${NC}"
final_hits=$(curl -s "http://localhost:${METRICS_PORT}/metrics" | grep "vllm:prefix_cache_hits_total{" | grep -oE '[0-9.]+$' || echo "0")
final_queries=$(curl -s "http://localhost:${METRICS_PORT}/metrics" | grep "vllm:prefix_cache_queries_total{" | grep -oE '[0-9.]+$' || echo "0")
echo -e "  Cache hits:    ${final_hits} (delta: +$(echo "$final_hits - $initial_hits" | bc))"
echo -e "  Cache queries: ${final_queries} (delta: +$(echo "$final_queries - $initial_queries" | bc))"
echo ""

# Calculate hit rate for this experiment
delta_hits=$(echo "$final_hits - $initial_hits" | bc)
delta_queries=$(echo "$final_queries - $initial_queries" | bc)
if [ "$delta_queries" != "0" ]; then
    hit_rate=$(echo "scale=1; $delta_hits * 100 / $delta_queries" | bc)
    echo -e "${GREEN}Experiment hit rate: ${hit_rate}%${NC}"
fi

# Cleanup
rm -f "$OBSERVER_LOG"

echo ""
echo -e "${CYAN}=========================================================${NC}"
echo -e "${CYAN}   Experiment Complete!${NC}"
echo -e "${CYAN}=========================================================${NC}"
echo ""
echo -e "Expected behavior:"
echo -e "  • Query A (1st): ${GREEN}📦 STORED${NC} - new block cached"
echo -e "  • Query B:       ${YELLOW}📦 STORED${NC} - new block cached (may evict old blocks)"
echo -e "  • Query C:       ${RED}📦 STORED${NC} - new block cached (may evict old blocks)"
echo -e "  • Query A (2nd): ${GREEN}✅ CACHE HIT${NC} - if A still in cache, or 📦 STORED if evicted"
echo ""
echo -e "With 16 blocks available, all 3 queries should fit without evicting each other."
echo -e "To force evictions, restart with: DYNAMO_NUM_GPU_BLOCKS_OVERRIDE=4"
echo ""


