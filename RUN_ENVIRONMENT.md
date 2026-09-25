# Local research run environment (2026-09-25)

The measured benchmark and model studies in `REPORT_EVIDENCE.md` were run on
Windows with a 12th Gen Intel Core i5-12500H (12 cores, 16 logical processors),
16.8 GB physical RAM, and no CUDA device. Python 3.11.7, PyTorch 2.7.0+cpu,
Transformers 4.51.3, sentence-transformers 4.1.0, and scikit-learn 1.5.2
were installed. The frozen test defense case latency includes rebuilding each
isolated attack index and therefore is not steady-state serving latency.

The optional local answer/SLM model is
`Qwen/Qwen2.5-0.5B-Instruct` at repository revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Its single
`model.safetensors` weight is 988,097,824 bytes with SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The weight and tokenizer/config files live in ignored `artifacts/models/` and
can be fetched with `python scripts/download_qwen_weight.py`. The model uses
CPU float weights in this environment; the slide's Llama-3-8B NF4 GPU profile
has not been run. Research generation requests are deterministic
(`do_sample=False`) and pass an explicit all-ones attention mask for unpadded
chat prompts.

The frozen first test run uses a cached
`sentence-transformers/all-MiniLM-L6-v2` encoder for retrieval, the
train-only strict sentence ranker, and the hashing research detector. Exact
artifact and code SHA-256 values are in `TEST_PROTOCOL_V1.md`, and result
fingerprints are in `TEST_EVIDENCE_V1.json`.
