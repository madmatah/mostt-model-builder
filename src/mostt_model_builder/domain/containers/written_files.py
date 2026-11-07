# Every step leaves its outputs in this mode, readable by the user's group, which the host
# runs as, since onnx writes its external data in 0600.
WRITTEN_FILE_MODE = 0o640
