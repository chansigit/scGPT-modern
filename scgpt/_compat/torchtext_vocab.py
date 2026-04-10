"""
Compatibility shim: a minimal pure-Python reimplementation of
``torchtext.vocab.Vocab`` and ``torchtext.vocab.vocab()``.

Why: ``torchtext`` was deprecated and unmaintained by the PyTorch team in 2024.
The last release (0.18.0) is pinned to torch 2.3 and its C++ extensions
(``libtorchtext.so``) fail to load with torch 2.5+ due to ABI breaks::

    OSError: libtorchtext.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs

scGPT only uses ``Vocab`` as a gene-name → integer-index lookup table with a
handful of methods (``__getitem__``, ``__contains__``, ``get_stoi``,
``insert_token``, ``append_token``, ``set_default_index``). Reimplementing
those in ~80 lines of pure Python is trivial and removes the torchtext
dependency entirely.

The API reproduced here is only what ``scgpt.tokenizer.gene_tokenizer`` actually
touches, plus the ``.vocab`` attribute that torchtext exposes to let its Vocab
be constructed from the inner C++ ``VocabPybind`` object (scGPT uses this
pattern in ``GeneVocab.__init__``).
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Union


class _VocabCore:
    """Inner container, mirrors the role of torchtext's C++ ``VocabPybind``.

    Holds the token list, the reverse lookup dict, and an optional default
    index that ``__getitem__`` returns for unknown tokens.
    """

    __slots__ = ("_itos", "_stoi", "_default_index")

    def __init__(self, tokens: Iterable[str]) -> None:
        self._itos: List[str] = list(tokens)
        self._stoi: Dict[str, int] = {t: i for i, t in enumerate(self._itos)}
        self._default_index: Optional[int] = None

    def __getitem__(self, token: str) -> int:
        idx = self._stoi.get(token)
        if idx is not None:
            return idx
        if self._default_index is not None:
            return self._default_index
        raise RuntimeError(
            f"Token {token!r} not in vocabulary and default index is not set"
        )

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._itos)


class Vocab:
    """Pure-Python replacement for ``torchtext.vocab.Vocab``.

    Can be constructed from any of:
      - ``None`` / no args → empty vocab
      - another :class:`Vocab`
      - a :class:`_VocabCore`
      - an :class:`OrderedDict` (tokens in insertion order)
      - a plain dict[str, int] mapping token → index

    The ``.vocab`` attribute returns the inner ``_VocabCore`` so that the
    torchtext idiom ``NewVocab(old_vocab.vocab)`` keeps working.
    """

    def __init__(
        self,
        vocab: Union["_VocabCore", "Vocab", Dict[str, int], OrderedDict, None] = None,
    ) -> None:
        if vocab is None:
            self.vocab = _VocabCore([])
        elif isinstance(vocab, _VocabCore):
            self.vocab = vocab
        elif isinstance(vocab, Vocab):
            # Share the inner core — matches torchtext behavior
            self.vocab = vocab.vocab
        elif isinstance(vocab, OrderedDict):
            self.vocab = _VocabCore(list(vocab.keys()))
        elif isinstance(vocab, dict):
            # Arbitrary dict — sort by value so indices match the dict values
            sorted_items = sorted(vocab.items(), key=lambda kv: kv[1])
            self.vocab = _VocabCore([t for t, _ in sorted_items])
        else:
            raise TypeError(
                f"Cannot construct Vocab from object of type {type(vocab).__name__}"
            )

    # --- lookup ------------------------------------------------------------

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, tokens: List[str]) -> List[int]:
        """``vocab(list_of_tokens)`` → list of indices."""
        return [self[t] for t in tokens]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[t] for t in tokens]

    def lookup_token(self, index: int) -> str:
        return self.vocab._itos[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        itos = self.vocab._itos
        return [itos[i] for i in indices]

    def get_stoi(self) -> Dict[str, int]:
        return dict(self.vocab._stoi)

    def get_itos(self) -> List[str]:
        return list(self.vocab._itos)

    # --- defaults ----------------------------------------------------------

    def set_default_index(self, idx: int) -> None:
        self.vocab._default_index = idx

    def get_default_index(self) -> Optional[int]:
        return self.vocab._default_index

    # --- mutation ----------------------------------------------------------

    def insert_token(self, token: str, index: int) -> None:
        if token in self.vocab:
            raise RuntimeError(f"Token {token!r} already exists in the vocabulary")
        if index < 0 or index > len(self.vocab):
            raise RuntimeError(
                f"Insert index {index} out of range [0, {len(self.vocab)}]"
            )
        self.vocab._itos.insert(index, token)
        # Rebuild stoi — cheap for scGPT's ~60k vocab
        self.vocab._stoi = {t: i for i, t in enumerate(self.vocab._itos)}

    def append_token(self, token: str) -> None:
        if token in self.vocab:
            raise RuntimeError(f"Token {token!r} already exists in the vocabulary")
        idx = len(self.vocab._itos)
        self.vocab._itos.append(token)
        self.vocab._stoi[token] = idx


# ---------------------------------------------------------------------------
# Module-level factory functions — torchtext ships these as top-level API.
# ---------------------------------------------------------------------------

def vocab(
    ordered_dict: "OrderedDict[str, int]",
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
) -> Vocab:
    """Replacement for ``torchtext.vocab.vocab``.

    Builds a :class:`Vocab` from an ordered ``{token: frequency}`` dict,
    keeping only tokens whose frequency ≥ ``min_freq``. Special tokens (if any)
    are prepended or appended per ``special_first``.
    """
    tokens = [tok for tok, freq in ordered_dict.items() if freq >= min_freq]
    if specials:
        # Deduplicate: specials take the reserved positions
        tokens = [t for t in tokens if t not in set(specials)]
        if special_first:
            tokens = list(specials) + tokens
        else:
            tokens = tokens + list(specials)
    return Vocab(_VocabCore(tokens))


def build_vocab_from_iterator(
    iterator: Iterable[Iterable[str]],
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
) -> Vocab:
    """Replacement for ``torchtext.vocab.build_vocab_from_iterator``.

    Consumes an iterable of token lists, counts frequencies, and builds a
    :class:`Vocab`. Not directly called by scGPT (it has its own variant in
    ``GeneVocab._build_vocab_from_iterator``) but included for completeness.
    """
    counter: Counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    sorted_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    kept = [tok for tok, freq in sorted_tuples if freq >= min_freq]

    if specials:
        kept = [t for t in kept if t not in set(specials)]
        if special_first:
            kept = list(specials) + kept
        else:
            kept = kept + list(specials)
    return Vocab(_VocabCore(kept))


__all__ = ["Vocab", "vocab", "build_vocab_from_iterator"]
