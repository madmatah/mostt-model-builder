REDACTED_SECRET = b"<redacted>"


class SecretRedactionError(ValueError):
    pass


# What a container prints arrives in chunks, and a secret may straddle two of them: the redaction
# keeps back the bytes that could begin an occurrence until the next chunk tells.
class SecretRedaction:
    def __init__(self, secret: bytes) -> None:
        if not secret:
            raise SecretRedactionError("an empty secret cannot be redacted")
        self.secret = secret
        self.pending = b""

    def redact(self, chunk: bytes) -> bytes:
        buffer = self.pending + chunk
        redacted = bytearray()
        start = 0
        while (index := buffer.find(self.secret, start)) != -1:
            redacted += buffer[start:index]
            redacted += REDACTED_SECRET
            start = index + len(self.secret)
        safe_end = max(start, len(buffer) - (len(self.secret) - 1))
        redacted += buffer[start:safe_end]
        self.pending = buffer[safe_end:]
        return bytes(redacted)

    def finish(self) -> bytes:
        rest = self.pending
        self.pending = b""
        return rest
