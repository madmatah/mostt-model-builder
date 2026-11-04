# Credits

mostt-model-builder reuses the work below. Each entry says what is reused, where it lives in this repository, where it comes from, and under which terms. License notices that must be kept are copied verbatim.

## Adapted code

- **parakeet-rs**, https://github.com/altunenes/parakeet-rs, at its commit `cadebd4469921ff426bf179e1cfd78fbe6fd9e05`: the export of Parakeet TDT to ONNX, ported from its `scripts/onnx_export.py` and `scripts/export_parakeet_tdt.py` into the export step of Parakeet TDT: the rules that map the names of Moondream's weights onto NeMo's (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/domain/export/weight_names.py`); the loading of Moondream's weights and the check of its tokenizer (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/steps/nemo_model.py`); `use_sdpa` and `exact_position_slices`, the dynamo export at opset 23 with its check of the dynamic axes, the export of the decoder and joint with its folding of constants, and the saving with external data (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/steps/onnx_export.py`); and the writing of `vocab.txt` (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/steps/nemo_export.py`). MIT OR Apache-2.0, taken under MIT, with this notice:

> MIT License
>
> Copyright (c) 2025 Enes Altun
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

- **NVIDIA-NeMo/Speech #16052**, https://github.com/NVIDIA-NeMo/Speech/pull/16052, merged as `4fcff72`: the forward of `ConformerConvolution`, which runs its convolutions of kernel size 1 as matrix products, ported onto NeMo 3.0.0 with the weights made 2-D before the export (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/steps/pointwise_forward.py`). Apache-2.0, under the notice of the file it changes, `nemo/collections/asr/parts/submodules/conformer_modules.py`:

> Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.

## Data

- **The names and shapes of the tensors** of Moondream's Parakeet Ultra, https://huggingface.co/moondream/parakeet-ultra, and of NVIDIA's Parakeet TDT 0.6B v3, https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3, as `describe` lists them, without any weight: `recipes/parakeet-tdt/tests/fixtures/moondream-tensors` and `recipes/parakeet-tdt/tests/fixtures/nemo-state-dict`, which test the mapping of the weights' names. Both models are under CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/: Parakeet Ultra by Moondream, and Parakeet TDT 0.6B v3 by NVIDIA.

## Libraries

- **uv** 0.13.7, by Astral, https://github.com/astral-sh/uv: its binary, copied from its image `ghcr.io/astral-sh/uv:0.13.7`, pinned by digest, into the `uv` image of the build (`images/uv/Containerfile`), which resolves the lock of the images' libraries. The images are never shipped. MIT OR Apache-2.0.
- **NeMo** 3.0.0 (`nemo-toolkit`, with its `asr` extra), by NVIDIA, https://github.com/NVIDIA-NeMo/NeMo: installed in the export image of Parakeet TDT (`recipes/parakeet-tdt/images/export/`), whose steps build the model from the configuration of a `.nemo` archive (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/steps/`). Apache-2.0.
- **PyTorch** 2.15.0 (`torch`), by the PyTorch Foundation, https://github.com/pytorch/pytorch: installed in the export image of Parakeet TDT, which runs NeMo's model on it, and in its Transformers image (`recipes/parakeet-tdt/images/transformers-reference/`), which runs Transformers' model on it. Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib, as its wheel declares.
- **safetensors** 0.8.0, by Hugging Face, https://github.com/huggingface/safetensors: installed in the export image of Parakeet TDT, which reads Moondream's weights in its format, and in its Transformers image, which reads them, and the tensors that NeMo's reference keeps, in that format. Apache-2.0.
- **ONNX** 1.23.2 (`onnx`), by the ONNX project, https://github.com/onnx/onnx, and **ONNX IR** 1.1.0 (`onnx-ir`), https://github.com/onnx/ir-py: installed in the export image of Parakeet TDT, for the export to ONNX, and in the core's `onnx-convert` image (`images/onnx-convert/`), for the conversions of the exported graph; ONNX alone also in the core's `onnx-inspect` image (`images/onnx-inspect/`), which reads the candidate's graphs without their weights. Apache-2.0.
- **ONNX Script** 0.7.2 (`onnxscript`), by Microsoft, https://github.com/microsoft/onnxscript: installed in the export image of Parakeet TDT, for the export to ONNX. MIT.
- **ONNX Runtime** 1.31.0 (`onnxruntime`, and `onnxruntime-gpu` with its `cuda` and `cudnn` extras), by Microsoft, https://github.com/microsoft/onnxruntime: installed in the export image of Parakeet TDT, which checks the exported graph on it, in the core's `onnx-convert` image, whose `MatMulNBitsQuantizer` converts the matrix products of a constant weight to 8-bit blocks (`src/mostt_model_builder/steps/onnx_conversion.py`), and in the transcription images of Parakeet TDT (`recipes/parakeet-tdt/images/transcribe-cpu/`, `recipes/parakeet-tdt/images/transcribe-gpu/`), which run onnx-asr on it. MIT.
- **onnxconverter-common** 1.16.0, by Microsoft, https://github.com/microsoft/onnxconverter-common: installed in the core's `onnx-convert` image, whose `convert_float_to_float16` converts the exported graph to float16, and to bfloat16 through it (`src/mostt_model_builder/steps/onnx_conversion.py`). MIT.
- **ml_dtypes** 0.6.0 (`ml-dtypes`), by the JAX authors, https://github.com/jax-ml/ml_dtypes: installed in the core's `onnx-convert` image, whose bfloat16 type the conversion to bfloat16 writes its weights in, and in the export image of Parakeet TDT through ONNX IR. Apache-2.0.
- **onnx-asr** 0.12.0, by Igor Stupakov, https://github.com/istupakov/onnx-asr: installed in the transcription images of Parakeet TDT, which transcribe FLEURS with the candidate's files under its names and its `config.json`. MIT.
- **NVIDIA CUDA libraries** (every `nvidia-*` package of `images/environment.lock`, at the versions it pins), by NVIDIA, https://developer.nvidia.com/cuda-toolkit: installed as tools in the GPU transcription image of Parakeet TDT through `onnxruntime-gpu[cuda,cudnn]`, and in its export and Transformers images through torch; the images are never shipped. Under NVIDIA's proprietary licenses, which `.claude/rules/configuration.md` > Dependencies accepts for the build images. With them, torch installs in the export and Transformers images `cuda-toolkit` 13.2.2, the meta-package that the same rule accepts, which declares no license, and NVIDIA's CUDA Python packages, https://github.com/NVIDIA/cuda-python, `cuda-bindings` 13.4.4 and `cuda-pathfinder` 1.8.5, under Apache-2.0, as their wheels declare.
- **NumPy** 2.5.4, by the NumPy developers, https://github.com/numpy/numpy: installed in the export image of Parakeet TDT, for the arrays of the export's checks, in its Transformers and transcription images, for the samples of the recordings, and in the core's `onnx-convert` image, for the weights that the conversions count and retype. BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0, as its wheel declares.
- **Transformers** 5.22.1 (`transformers`), by Hugging Face, https://github.com/huggingface/transformers: installed in the Transformers image of Parakeet TDT, whose step loads Moondream's weights in its `ParakeetForTDT` and v3's processor, to check them against NeMo's reference and transcribe FLEURS fr (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/steps/transformers_reference.py`). Apache-2.0.
- **librosa** 1.0.0, by the librosa development team, https://github.com/librosa/librosa: installed in the Transformers image of Parakeet TDT, for the feature extractor of Transformers' processor. ISC.
- **ANTLR 4 Python runtime** 4.9.3 (`antlr4-python3-runtime`), by the ANTLR project, https://github.com/antlr/antlr4: built from its sdist, pinned by hash, in a stage of the export image of Parakeet TDT (`recipes/parakeet-tdt/images/export/Containerfile`), since hydra-core and omegaconf require it for NeMo and it publishes no wheel. BSD-3-Clause.
- **setuptools** 84.2.1, by the Python Packaging Authority, https://github.com/pypa/setuptools: downloaded as a wheel, pinned by hash (`recipes/parakeet-tdt/images/export/antlr4-build-requirements.txt`), into the build stage of the export image of Parakeet TDT, which builds antlr4-python3-runtime with it; the image itself also installs setuptools, at the version that `images/environment.lock` pins, as a dependency of torch and NeMo, and so does the Transformers image of Parakeet TDT, as a dependency of torch. MIT.
