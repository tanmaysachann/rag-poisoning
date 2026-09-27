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

The 2026-09-27 defender-aware PPO comparison ran three 200-episode CPU
training seeds (42, 43, 44) against the frozen hashing detector and the same
75 validation questions per seed. Fixed substitution and random edits used
the same wrong-answer schedule for each seed. The raw checkpoints and case
files are hash-recorded in `results/ppo_multiseed_validation.json`.
`python scripts/verify_release.py` checks the saved comparison, the frozen v1
evidence, v2 structure and hashes, the full test suite, and an isolated demo
rebuild. It does not evaluate the v2 test split.

The detector-risk reward ablation reused the same CPU, frozen hashing detector,
three 200-episode seeds, and 75 validation questions per seed. The only changed
training reward weight was detection shaping, from 0.5 to zero; terminal
defended-success reward remained active. Both raw result sets and checkpoint
hashes are recorded in `results/ppo_detection_reward_ablation_validation.json`.

The Contriever-msmarco CPU retrieval comparison used the official Meta model
revision `abe8c1493371369031bcb1e02acb754cf4e162fa`, with a 438,007,537
byte `pytorch_model.bin` whose SHA-256 is
`08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8`.
It used four PyTorch CPU threads, attention-mask mean pooling, L2 normalized
768-dimensional vectors, and 256-token maximum inputs. Only v1 validation
questions and passages were scored; v2 test stayed untouched.
