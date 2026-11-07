---
license: "cc-by-4.0"
language: ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"]
pipeline_tag: "automatic-speech-recognition"
tags: ["onnx", "onnx-asr", "parakeet", "tdt", "speech-recognition"]
base_model: "moondream/parakeet-ultra"
base_model_relation: "quantized"
---

# Parakeet Ultra in ONNX

Moondream's Parakeet Ultra, a post-training of NVIDIA's Parakeet TDT 0.6B v3, exported to ONNX in the layout of onnx-asr, with its encoder in float32, float16, bfloat16 and 8-bit blocks.

## Languages

The model transcribes 25 languages and detects the language by itself: Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, Ukrainian.

This card measures English, Latin American Spanish, French, Brazilian Portuguese and German on [FLEURS](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd); the other languages are those NVIDIA's Parakeet TDT 0.6B v3 declares, not measured here.

## Variants

The variants share the same weights and differ in the precision of the encoder. Pick float32 for reference accuracy.

### float32 (FP32)

The reference: the exported model, unconverted, against which the other variants are measured.

- **Runs on:** CPU, GPU
- **Size:** 0.00 GB

| Language | WER | Against the original model in NeMo, CPU |
|---|---|---|
| English (en_us) | 4.50 % | -0.01 points |
| Latin American Spanish (es_419) | 4.50 % | -0.01 points |
| French (fr_fr) | 4.50 % | -0.01 points |
| Brazilian Portuguese (pt_br) | 4.50 % | -0.01 points |
| German (de_de) | 4.50 % | -0.01 points |

Files:

- `encoder-model.onnx`
- `encoder-model.onnx.data`
- `decoder_joint-model.onnx`
- `vocab.txt`, shared by every variant
- `config.json`, shared by every variant

### float16 (FP16)

The encoder converted to float16, for the GPU.

- **Runs on:** GPU; on the CPU, ONNX Runtime computes most of it in float32 again
- **Size:** 0.00 GB

| Language | WER | Against float32, GPU |
|---|---|---|
| English (en_us) | 4.50 % | +0.01 points |
| Latin American Spanish (es_419) | 4.50 % | +0.01 points |
| French (fr_fr) | 4.50 % | +0.01 points |
| Brazilian Portuguese (pt_br) | 4.50 % | +0.01 points |
| German (de_de) | 4.50 % | +0.01 points |

Files:

- `encoder-model.fp16.onnx`
- `decoder_joint-model.fp16.onnx`
- `vocab.txt`, shared by every variant
- `config.json`, shared by every variant

### bfloat16 (BF16)

The encoder converted to bfloat16, with Pad, Tile and Floor kept in float32, for the GPU.

- **Runs on:** GPU; on the CPU, ONNX Runtime computes most of it in float32 again
- **Size:** 0.00 GB

| Language | WER | Against float32, GPU |
|---|---|---|
| English (en_us) | 4.50 % | +0.01 points |
| Latin American Spanish (es_419) | 4.50 % | +0.01 points |
| French (fr_fr) | 4.50 % | +0.01 points |
| Brazilian Portuguese (pt_br) | 4.50 % | +0.01 points |
| German (de_de) | 4.50 % | +0.01 points |

Files:

- `encoder-model.bf16.onnx`
- `decoder_joint-model.bf16.onnx`
- `vocab.txt`, shared by every variant
- `config.json`, shared by every variant

### int8

Recommended on the CPU: the smallest variant. The encoder is in 8-bit blocks of 64 symmetric weights.

- **Runs on:** CPU, GPU
- **Size:** 0.00 GB

| Language | WER | Against float32, CPU |
|---|---|---|
| English (en_us) | 4.50 % | +0.01 points |
| Latin American Spanish (es_419) | 4.50 % | +0.01 points |
| French (fr_fr) | 4.50 % | +0.01 points |
| Brazilian Portuguese (pt_br) | 4.50 % | +0.01 points |
| German (de_de) | 4.50 % | +0.01 points |

Files:

- `encoder-model.int8.onnx`
- `decoder_joint-model.int8.onnx`
- `vocab.txt`, shared by every variant
- `config.json`, shared by every variant

The rates and comparisons are measured as [How the rates were measured](#how-the-rates-were-measured) describes.

## How the rates were measured

The word error rate (WER) counts the words a transcript gets wrong (substituted, missing or added) against FLEURS's reference, over the number of words in the reference. Each variant transcribed the test sets of FLEURS (350 utterances in English, 350 in Latin American Spanish, 350 in French, 350 in Brazilian Portuguese, 350 in German) greedily, one utterance at a time, on each device it runs on, with onnx-asr 0.12.0 and onnxruntime 1.31.0 on the CPU, and with onnx-asr 0.12.0 and onnxruntime-gpu 1.31.0 on the GPU. The rate shown is the one measured on the CPU, or on the GPU for the variants that only run there: the runtime and the hardware, not the files, make the two differ slightly.

Before counting errors, both the transcript and FLEURS's reference are normalized the same way: Unicode NFKC, lowercase, punctuation removed, typographic apostrophes read as plain ones.

The `<unk>` tokens are removed from the transcripts. The vocabulary of Parakeet TDT has no piece for some characters FLEURS writes, such as « or ’, so the model writes `<unk>` where they stand; counting them as errors would penalize the vocabulary, not the recognition.

Each variant is compared with its reference on the same utterances: float32 with the original model in NeMo, the other variants with float32 on the same device. float32 must give the same tokens as NeMo on every utterance, and a variant is accepted only if it loses at most 0.2 points against its reference, at the upper end of a 95 % interval from a paired bootstrap.

## Limitations

Moondream's voice activity head, which its own runtime uses to cut long recordings at their pauses, is not exported.

## How the files were built

1. NeMo builds the model from the configuration of Parakeet TDT 0.6B v3 alone, without loading its checkpoint, and loads the weights of Parakeet Ultra into it.
2. The export writes the ONNX files, then checks the exported encoder against NeMo's on the first 20 utterances of FLEURS fr.
3. NeMo transcribes English, Latin American Spanish, French, Brazilian Portuguese and German from FLEURS in float32 on the CPU: this run of the original model in NeMo is the reference of the comparisons.
4. Transformers reads the weights of Parakeet Ultra as Moondream publishes them, and its encoder and joint are checked against NeMo's, so that the mapping of the weights is checked from a second side.
5. onnx-asr transcribes English, Latin American Spanish, French, Brazilian Portuguese and German from FLEURS with every variant, on each device that its section under Variants gives.

Every file was copied out of the container that wrote it, and refused if it began as code, a pickle or an archive, or broke the structure of its format. Every ONNX graph was inspected against the build's rules, on its nodes, its tensors and its quantized products. The verification then measured every run on FLEURS, and held every comparison of two runs and every parity of an export with its reference framework to its rule.

Every step ran in a container without network, from pinned sources, libraries and base images.

The code of this build is [mostt-model-builder](https://github.com/madmatah/mostt-model-builder).

### Sources

- [google/fleurs](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
- [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra/tree/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b)
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b)

## Usage

The files follow onnx-asr's layout for a NeMo TDT model: config.json selects its preprocessor, and the suffix of the encoder and of the decoder and joint names the variant: none for float32, fp16 for float16, bf16 for bfloat16 and int8 for 8-bit blocks.

onnx-asr loads the directory of these files as a NeMo TDT model, given the quantization of the variant to run.

encoder-model.onnx reads its weights from encoder-model.onnx.data, which lies beside it; each other encoder is a single file.

## License and attribution

| Work | Creator | Revision | License |
|---|---|---|---|
| [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra/tree/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b) | Moondream | `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b) | NVIDIA | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Modifications:

- Parakeet Ultra adapts Parakeet TDT 0.6B v3: Moondream post-trained its weights and kept its architecture and its tokenizer.
- The weights of Parakeet Ultra, without Moondream's voice activity head, are loaded into NeMo's model of Parakeet TDT 0.6B v3 and exported to ONNX in float32: the encoder, with its external data, and the decoder and joint of one prediction step.
- In the exported encoder, the 48 pointwise convolutions run as matrix products and the attention of each layer as one Attention node.
- The encoder is converted to float16; to bfloat16, with Pad, Tile and Floor kept in float32; and to 8-bit blocks of 64 symmetric weights. The decoder and joint stay in float32 for every variant.
- The vocabulary is that of Parakeet TDT 0.6B v3, one piece and its id per line.

These files are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are provided as is, without warranties, as section 5 of the license states. Neither Moondream nor NVIDIA endorses these files.
