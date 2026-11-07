---
license: "cc-by-4.0"
language: ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"]
pipeline_tag: "automatic-speech-recognition"
tags: ["onnx", "onnx-asr", "parakeet", "tdt", "speech-recognition"]
base_model: "moondream/parakeet-ultra"
base_model_relation: "quantized"
---

# $title

Moondream's Parakeet Ultra, a post-training of NVIDIA's Parakeet TDT 0.6B v3, exported to ONNX in the layout of onnx-asr, with its encoder in float32, float16, bfloat16 and 8-bit blocks.

## Variants

Word error rates on the test sets of FLEURS, without the unknown tokens, which the vocabulary cannot write.

$variant_table

Each variant was transcribed on every device it lists, and each of those runs passed its comparisons. A variant's rates are those of its run on the CPU, or on the GPU when it was not run on the CPU: the runtime and the hardware, not the files, make the devices differ. A size counts every file the variant loads, the files of every variant included.

$variant_sections

### Files of every variant

$shared_files.

## Languages

The model declares 25 languages: en, es, fr, de, bg, hr, cs, da, nl, et, fi, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, sv, ru, uk. [FLEURS](https://huggingface.co/datasets/google/fleurs/tree/$fleurs_revision) measured $measured_languages only: the other languages are declared, not measured here.

## Limitations

FLEURS measured French and English alone; the other declared languages are those of Parakeet TDT 0.6B v3, unmeasured here.

Moondream's voice activity head, which its own runtime uses to cut long recordings at their pauses, is not exported.

## How the files were built

1. NeMo builds the model from the configuration of Parakeet TDT 0.6B v3 alone, without loading its checkpoint, and loads the weights of Parakeet Ultra into it.
2. The export writes the ONNX files, then checks the exported encoder against NeMo's on the first 20 utterances of FLEURS fr.
3. NeMo transcribes FLEURS fr and en in float32 on the CPU: this run of the original model in NeMo is the reference of the comparisons.
4. Transformers reads the weights of Parakeet Ultra as Moondream publishes them, and its encoder and joint are checked against NeMo's, so that the mapping of the weights is checked from a second side.
5. onnx-asr transcribes FLEURS fr and en with every variant, on each device that the table of the variants gives it.

Every file was copied out of the container that wrote it, and refused if it began as code, a pickle or an archive, or broke the structure of its format. Every ONNX graph was inspected against the build's rules, on its nodes, its tensors and its quantized products. The verification then measured every run on FLEURS, and held every comparison of two runs and every parity of an export with its reference framework to its rule.

Every step ran in a container without network, from pinned sources, libraries and base images.

The code of this build is [mostt-model-builder](https://github.com/madmatah/mostt-model-builder).

### Sources

$sources

## Usage

The files follow onnx-asr's layout for a NeMo TDT model: config.json selects its preprocessor, and the suffix of the encoder and of the decoder and joint names the variant: none for float32, fp16 for float16, bf16 for bfloat16 and int8 for 8-bit blocks.

onnx-asr loads the directory of these files as a NeMo TDT model, given the quantization of the variant to run.

encoder-model.onnx reads its weights from encoder-model.onnx.data, which lies beside it; each other encoder is a single file.

The rates above were measured $measured_with.

## License and attribution

| Work | Creator | Revision | License |
|---|---|---|---|
| [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra/tree/$ultra_revision) | Moondream | `$ultra_revision` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/$v3_revision) | NVIDIA | `$v3_revision` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Modifications:

- Parakeet Ultra adapts Parakeet TDT 0.6B v3: Moondream post-trained its weights and kept its architecture and its tokenizer.
- The weights of Parakeet Ultra, without Moondream's voice activity head, are loaded into NeMo's model of Parakeet TDT 0.6B v3 and exported to ONNX in float32: the encoder, with its external data, and the decoder and joint of one prediction step.
- In the exported encoder, the 48 pointwise convolutions run as matrix products and the attention of each layer as one Attention node.
- The encoder is converted to float16; to bfloat16, with Pad, Tile and Floor kept in float32; and to 8-bit blocks of 64 symmetric weights. The decoder and joint stay in float32 for every variant.
- The vocabulary is that of Parakeet TDT 0.6B v3, one piece and its id per line.

These files are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are provided as is, without warranties, as section 5 of the license states. Neither Moondream nor NVIDIA endorses these files.
