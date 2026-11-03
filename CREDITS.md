# Credits

mostt-model-builder reuses the work below. Each entry says what is reused, where it lives in this repository, where it comes from, and under which terms. License notices that must be kept are copied verbatim.

## Adapted code

- **mostt's model store**, at its commit `920e03b`: the suffixes that mark code, a pickle, an archive or a format that runs code when it loads, and the check of every suffix of a file name whatever its case, adapted from `REFUSED_SUFFIXES` and `refused_suffixes_of` in `server/src/mostt_server/domain/catalogue.py`, with the cases of its tests, so that every file this project publishes passes the store that installs it (`src/mostt_model_builder/domain/formats/file_names.py`). MIT, under this notice:

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

None yet.
