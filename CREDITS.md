# Credits

mostt-model-builder reuses the work below. Each entry says what is reused, where it lives in this repository, where it comes from, and under which terms. License notices that must be kept are copied verbatim.

## Adapted code

- **mostt's model store**, at its commit `920e03b`: the grammar of a file name, `FILE_NAME_GRAMMAR`, the suffixes that mark code, a pickle, an archive or a format that runs code when it loads, and the check of every suffix of a file name whatever its case, adapted from `FILE_NAME_GRAMMAR`, `REFUSED_SUFFIXES` and `refused_suffixes_of` in `server/src/mostt_server/domain/catalogue.py`, with the cases of its tests, so that every file this project publishes passes the store that installs it (`src/mostt_model_builder/domain/formats/file_names.py`). MIT, under this notice:

> MIT License
>
> Copyright (c) 2026 The mostt authors
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

## Libraries

- **uv** 0.13.7, by Astral, https://github.com/astral-sh/uv: its binary, copied from its image `ghcr.io/astral-sh/uv:0.13.7`, pinned by digest, into the `uv` image of the build (`images/uv/Containerfile`), which resolves the lock of the images' libraries. The images are never shipped. MIT OR Apache-2.0.
- **NeMo** 3.0.0 (`nemo-toolkit`, with its `asr` extra), by NVIDIA, https://github.com/NVIDIA-NeMo/NeMo: installed in the export image of Parakeet TDT (`recipes/parakeet-tdt/images/export/`), whose steps build the model from the configuration of a `.nemo` archive (`recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/steps/`). Apache-2.0.
- **PyTorch** 2.15.0 (`torch`), by the PyTorch Foundation, https://github.com/pytorch/pytorch: installed in the export image of Parakeet TDT, which runs NeMo's model on it. Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib, as its wheel declares.
- **safetensors** 0.8.0, by Hugging Face, https://github.com/huggingface/safetensors: installed in the export image of Parakeet TDT, which reads Moondream's weights in its format. Apache-2.0.
- **ONNX** 1.23.2 (`onnx`), by the ONNX project, https://github.com/onnx/onnx, and **ONNX IR** 1.1.0 (`onnx-ir`), https://github.com/onnx/ir-py: installed in the export image of Parakeet TDT, for the export to ONNX. Apache-2.0.
- **ONNX Script** 0.7.2 (`onnxscript`), by Microsoft, https://github.com/microsoft/onnxscript: installed in the export image of Parakeet TDT, for the export to ONNX. MIT.
- **ONNX Runtime** 1.31.0 (`onnxruntime`), by Microsoft, https://github.com/microsoft/onnxruntime: installed in the export image of Parakeet TDT, which checks the exported graph on it. MIT.
- **NumPy** 2.5.4, by the NumPy developers, https://github.com/numpy/numpy: installed in the export image of Parakeet TDT, for the arrays of the export's checks. BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0, as its wheel declares.
- **ANTLR 4 Python runtime** 4.9.3 (`antlr4-python3-runtime`), by the ANTLR project, https://github.com/antlr/antlr4: built from its sdist, pinned by hash, in a stage of the export image of Parakeet TDT (`recipes/parakeet-tdt/images/export/Containerfile`), since hydra-core and omegaconf require it for NeMo and it publishes no wheel. BSD-3-Clause.
- **setuptools** 84.2.1, by the Python Packaging Authority, https://github.com/pypa/setuptools: downloaded as a wheel, pinned by hash (`recipes/parakeet-tdt/images/export/antlr4-build-requirements.txt`), into the build stage of the export image of Parakeet TDT, which builds antlr4-python3-runtime with it; the image itself also installs setuptools, at the version that `images/uv.lock` pins, as a dependency of torch and NeMo. MIT.
