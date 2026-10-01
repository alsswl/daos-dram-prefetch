# Synthetic Multi-Round QA Benchmark

This benchmark tool generates synthetic multi-round conversations and measures the performance of LLM serving systems.

## Configuration

The benchmark supports two main scenarios:

1. **Short Input, Short Output**
   - System prompt: 0 tokens
   - Chat history: 256 tokens
   - Answer length: 20 tokens
   - Default QPS: 15

2. **Long Input, Short Output**
   - System prompt: 1000 tokens
   - Chat history: 20000 tokens
   - Answer length: 100 tokens
   - Default QPS: 0.1

## Running the Benchmark

### Short Input, Short Output

```bash
# Run with default QPS (15)
./short_input_short_output.sh <model> <base_url> <save_file_key>

# Run with multiple QPS values
./short_input_short_output.sh <model> <base_url> <save_file_key> 15 20 25
```

### Long Input, Short Output

```bash
# Run with default QPS (0.1)
./long_input_short_output_run.sh <model> <base_url> <save_file_key>

# Run with multiple QPS values
./long_input_short_output_run.sh <model> <base_url> <save_file_key> 0.1 0.2 0.3
```

### Warm-up Phase

For the long input scenario, you can run a warm-up phase separately:

```bash
./long_input_short_output_warmup.sh <model> <base_url>
```

## Processing Results

To calculate the average TTFT (Time To First Token):

```bash
python3 multi-round-qa.py --process-summary <your_csv_file>
```

To calculate the average ITL (Inter-Token Latency):

```bash
python3 calculat_itl.py
```

## Notes

- The warm-up phase preloads the KV cache for better performance measurement
- The benchmark automatically handles the correct script paths regardless of where it's run from
- QPS values can be customized through command-line arguments
- Results are saved in CSV format with the QPS value in the filename
