"""Air-gap GGUF packaging notes

1. Train + merge to `checkpoints/sei-8b-instruct`
2. `python scripts/export_gguf.py --model checkpoints/sei-8b-instruct --quant Q4_K_M --llama-cpp /path/to/llama.cpp`
3. Bundle contents (copied by export script into `checkpoints/gguf/`):
   - `*.gguf`
   - `sei_v1.json`
   - `sei_v1.gbnf` (or `sei_v1_enum.gbnf` from `scripts/build_grammar.py`)
   - `technique_ids.txt`
   - `NOTICE`
4. Serve offline:
   ```bash
   llama-cli -m sei-8b-instruct-q4_k_m.gguf --grammar-file sei_v1.gbnf -n 1024 -p "..."
   ```
"""
