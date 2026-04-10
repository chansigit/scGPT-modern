"""Compatibility shims for scGPT.

Currently provides a drop-in replacement for the old flash-attn v1 ``FlashMHA``
class, backed by modern flash-attn-4 (CuTeDSL) or flash-attn v2 functional API.
"""
