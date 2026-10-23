# strata_model_prep.py

A helper script for [Strata](https://github.com/Niko1221/Strata), the engine that runs Qwen3.8-Flash-Next on a normal PC
(one GPU plus system RAM).

You download a model (GGUF files) yourself. The script turns it into the files Strata needs, and writes the JSON config
that starts it with `serve/server.py`. It uses Strata's own tools and follows the same rules as Strata's `setup.py`, so
the result is the same as a setup install.

## What it does

1. **Checks the model.** It finds every shard of the GGUF, checks that each file is complete (its size matches what its
   header says), and finds the shard that holds the n-gram table (`per_layer_token_embd.weight`).
2. **Builds the pack** with `tools/iq_pack.py` (a few seconds to a minute). It adds `--compat-bf16` only when the model
   needs it.
3. **Builds the MTP draft layer** (speculative decoding) with `tools/mtp_fetch.py`, `tools/mtp_pack.py` and
   `tools/mtp_rt.py`. If another model already has one, it copies it instead of downloading ~5 GB again.
4. **Marks the model files as done.** It writes a `.done` file next to each GGUF, like `setup.py` does.
5. **Writes the JSON config** for `serve/server.py`, plus a `run-<model>.sh` launch script.
6. **Checks the config** (optional). It confirms that every path exists and that the engine accepts every option, then
   starts the server on a test port. With `--verify-load` it loads the model for real.

## One folder per model

Each model gets its own folder of artifacts, so several models can live side by side without sharing any file:

```
Strata-data/
├── models/<download folder>/          the GGUF files you downloaded (+ .done marks)
└── artifacts/<model>/
    ├── model-config-source.json       which GGUF this folder belongs to
    ├── packs/                         the pack (--pack) and its tokenizer/
    └── mtp/                           the MTP tensors
        └── rt/                        the MTP draft layer (--mtp)
```

The folders are created when they are missing. Every file is a real copy (no links). If two models have GGUF files with
the same name, the second one gets `<download folder>__<model>` as its folder name.

## Requirements

- A working Strata install (the folder with `serve/server.py`, `tools/`, `data/` and `engine/strata`), set up once with
  `./setup.sh`.
- Python 3.8 or newer. Running it with Strata's own `.venv/bin/python` is recommended.
- Qwen3.8-Flash-Next GGUF files: the original, Swift 1.5, the Coder, Unsloth's quants, or fine-tunes of them.

## Usage

Put the script in the Strata folder and run it from there.

Build everything and write the config, answering the questions as they come:

```bash
./strata_model_prep.py --convert-model --generate-model-file
```

The same, with no questions (MTP built if missing, 128K context):

```bash
./strata_model_prep.py --convert-model --build-mtp --generate-model-file --yes \
    --model-dir /path/to/Strata-data/models/<download folder>/<model>-00001-of-00002.gguf \
    --context 131072
```

Write only the config, for a model that is already built:

```bash
./strata_model_prep.py --generate-model-file --model-dir /path/to/Strata-data/models/<download folder>
```

Check a config, then load the model with it:

```bash
./strata_model_prep.py --verify --verify-load --config-out <model>.json
```

Start the model:

```bash
.venv/bin/python serve/server.py --engine strata --config <model>.json --port 8080
```

`--model-dir` takes any shard, the model's folder, or its pack. If a folder holds several models (for example several
quants), the script asks which one.

## Main options

| Option | What it does |
|---|---|
| `--convert-model` | Build the pack and the MTP draft layer. |
| `--generate-model-file` | Write the JSON config. |
| `--verify`, `--verify-load` | Check a config; also load the model. |
| `--model-dir` | The downloaded model (a shard, its folder or its pack). |
| `--artifacts-dir` | The artifacts folder (default: `<Strata-data>/artifacts`). |
| `--config-out` | Where to write the JSON (default: the Strata folder). |
| `--context` | Context length in tokens. Past 262144, YaRN rope scaling is added. |
| `--kv-quant` | KV cache precision: `int8` (default), `q4_0`, `k8v4` or `fp16`. |
| `--kv-resident`, `--kv-streaming` | KV streaming (on by default from 64K when the RAM holds it). |
| `--build-mtp` | Download and build the MTP draft layer when there is none to copy. |
| `--draft-vocab` | The draft layer's token subset: `cjk` (default), `en`, `cyrillic` or `fr`. |
| `--vision` | Images: `auto`, `on`, `cpu` or `off` (needs an engine built with images). |
| `--low-ram` + `--experts-bin` | Strata's low-RAM mode. |
| `--yes`, `--overwrite` | Run without questions; replace existing files. |
| `--suggestions` | Explain the main performance settings. |

Run `./strata_model_prep.py --help` for the full list.

## What the config gets

The engine options follow `setup.py`:

- the pack, the model's shard 1 and the n-gram table's shard (left out for files of more than two shards, where the
  engine finds it itself);
- the expert profile that matches the model's expert count (512, or 256 for the Coder);
- `--expert-cache auto`, `--prefill auto`, `--spec 4`, `--spec-min-p 0.5` and the model's own MTP draft layer;
- the context, KV precision and KV streaming;
- CPU expert workers on hybrid Intel CPUs (a saved `./setup.sh --calibrate` result is used instead when there is one);
- for Unsloth's files, the engine settings `setup.py` writes for them.

When a config of the same name already exists, it is kept as `<model>.json.bak`. Its API model name, its log file and
any keys you added by hand are carried over.

## Notes

- Everything runs on your PC. The only download is the MTP draft layer (~5 GB, from the original Qwen checkpoint), and
  only when you ask for it.
- Each model's artifacts take about 8 GB (a 1.5 GB pack and a 6.5 GB MTP folder), next to the model's own GGUF files.
- Tested on Linux with an NVIDIA RTX PRO 5000 (24 GB) and 123 GB of RAM. AMD (HIP) and Windows are not tested.
