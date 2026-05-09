"""Tokenizer for MOIRAIS's inference engine.

Loads tokenization data from GGUF model metadata (the ``tokenizer.ggml.*``
keys) and provides encode/decode without requiring SentencePiece at runtime.

Falls back to SentencePiece if a ``.model`` file is provided explicitly.
"""

from __future__ import annotations

from pathlib import Path


class Tokenizer:
    """BPE / SentencePiece tokenizer loaded from GGUF metadata.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a GGUF file (reads ``tokenizer.ggml.*`` metadata) or a
        SentencePiece ``.model`` file.
    gguf_model : GGUFModel, optional
        An already-loaded :class:`moirais.gguf_loader.GGUFModel` instance.

    Examples
    --------
    >>> from moirais.gguf_loader import GGUFModel
    >>> model = GGUFModel("path/to/model.gguf")
    >>> tok = Tokenizer(gguf_model=model)
    >>> ids = tok.encode("Hello world")
    >>> tok.decode(ids)
    'Hello world'
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        gguf_model=None,
    ):
        self._vocab: list[str] = []
        self._scores: list[float] = []
        self._token_to_id: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []
        self._eos_id: int = 2
        self._bos_id: int = 1
        self._sp = None  # SentencePiece processor (optional fallback)

        if gguf_model is not None:
            self._load_from_gguf(gguf_model)
        elif model_path is not None:
            path = Path(model_path)
            if path.suffix == ".model":
                self._load_sentencepiece(path)
            else:
                from .gguf_loader import GGUFModel

                gm = GGUFModel(path)
                self._load_from_gguf(gm)
        else:
            raise ValueError("Provide either model_path or gguf_model")

    def _load_from_gguf(self, gm) -> None:
        """Extract tokenizer from GGUF metadata keys."""
        meta = gm._metadata

        # Vocabulary tokens
        tokens = meta.get("tokenizer.ggml.tokens", [])
        if not tokens:
            raise ValueError("GGUF file has no tokenizer.ggml.tokens metadata")

        self._vocab = [t if isinstance(t, str) else t.decode("utf-8", errors="replace") for t in tokens]
        self._scores = meta.get("tokenizer.ggml.scores", [0.0] * len(self._vocab))
        self._token_to_id = {tok: i for i, tok in enumerate(self._vocab)}

        # Special tokens
        self._bos_id = meta.get("tokenizer.ggml.bos_token_id", 1)
        self._eos_id = meta.get("tokenizer.ggml.eos_token_id", 2)

        # BPE merges (if present)
        merges_raw = meta.get("tokenizer.ggml.merges", [])
        self._merges = []
        for m in merges_raw:
            parts = m.split(" ", 1) if isinstance(m, str) else []
            if len(parts) == 2:
                self._merges.append((parts[0], parts[1]))

    def _load_sentencepiece(self, path: Path) -> None:
        """Load from a SentencePiece .model file."""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece is required for .model files: pip install sentencepiece")
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(path))
        self._vocab = [self._sp.IdToPiece(i) for i in range(self._sp.GetPieceSize())]
        self._token_to_id = {tok: i for i, tok in enumerate(self._vocab)}
        self._bos_id = self._sp.bos_id()
        self._eos_id = self._sp.eos_id()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        """Encode text to token IDs.

        Uses SentencePiece if available, otherwise greedy BPE matching.
        """
        if self._sp is not None:
            ids = self._sp.Encode(text)
            if add_bos and (not ids or ids[0] != self._bos_id):
                ids = [self._bos_id] + ids
            return ids

        # Greedy byte-fallback encoding for GGUF vocab
        tokens = self._bpe_encode(text)
        ids = [self._token_to_id.get(t, 0) for t in tokens]
        if add_bos:
            ids = [self._bos_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if self._sp is not None:
            return self._sp.Decode(ids)

        pieces = []
        for i in ids:
            if 0 <= i < len(self._vocab):
                piece = self._vocab[i]
                # SentencePiece-style: ▁ means space
                piece = piece.replace("▁", " ")
                # Byte tokens: <0xHH>
                if piece.startswith("<0x") and piece.endswith(">"):
                    try:
                        pieces.append(chr(int(piece[3:-1], 16)))
                        continue
                    except ValueError:
                        pass
                pieces.append(piece)
        text = "".join(pieces)
        # Strip leading space if BOS was decoded
        if text.startswith(" "):
            text = text[1:]
        return text

    def _bpe_encode(self, text: str) -> list[str]:
        """Greedy longest-match encoding with UTF-8 byte fallback."""
        # SentencePiece convention: leading space becomes ▁
        text = "▁" + text.replace(" ", "▁")
        tokens = []
        i = 0
        while i < len(text):
            # Greedy: find longest vocab match starting at position i
            best = None
            best_len = 0
            for length in range(min(32, len(text) - i), 0, -1):
                candidate = text[i : i + length]
                if candidate in self._token_to_id:
                    best = candidate
                    best_len = length
                    break
            if best is not None:
                tokens.append(best)
                i += best_len
            else:
                # Byte fallback: encode character as UTF-8 byte tokens
                char_bytes = text[i].encode("utf-8")
                for b in char_bytes:
                    byte_tok = f"<0x{b:02X}>"
                    tokens.append(byte_tok)
                i += 1
        return tokens

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        if self._sp is not None:
            return self._sp.GetPieceSize()
        return len(self._vocab)

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self._eos_id

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence token ID."""
        return self._bos_id

    def __repr__(self) -> str:
        src = "sentencepiece" if self._sp else "gguf"
        return f"Tokenizer(vocab_size={self.vocab_size}, source={src})"
