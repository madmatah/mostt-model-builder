ULTRA_REPOSITORY = "moondream/parakeet-ultra"
ULTRA_CONFIGURATION = "config.json"
ULTRA_TOKENIZER = "tokenizer.json"
ULTRA_WEIGHTS = "model.safetensors"
NEMO_REPOSITORY = "nvidia/parakeet-tdt-0.6b-v3"
NEMO_ARCHIVE = "parakeet-tdt-0.6b-v3.nemo"
PROCESSOR_CONFIGURATION = "processor_config.json"
TOKENIZER_CONFIGURATION = "tokenizer_config.json"
GENERATION_CONFIGURATION = "generation_config.json"
# Transformers loads a model and its processor from one directory, where Moondream publishes the
# weights and v3 the processor.
ULTRA_FILES = (ULTRA_CONFIGURATION, ULTRA_TOKENIZER, ULTRA_WEIGHTS)
PROCESSOR_FILES = (PROCESSOR_CONFIGURATION, TOKENIZER_CONFIGURATION, GENERATION_CONFIGURATION)
