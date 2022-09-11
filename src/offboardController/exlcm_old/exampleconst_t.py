"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class exampleconst_t(object):
    __slots__ = []

    __typenames__ = []

    __dimensions__ = []

    ABC = 1
    DEF = 2
    PI = 3.1415926
    E = 2.8718
    LONG = 0xf0f0f0f0

    def __init__(self):
        pass

    def encode(self):
        buf = BytesIO()
        buf.write(exampleconst_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        pass

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != exampleconst_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return exampleconst_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = exampleconst_t()
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if exampleconst_t in parents: return 0
        tmphash = (0x12345678) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if exampleconst_t._packed_fingerprint is None:
            exampleconst_t._packed_fingerprint = struct.pack(">Q", exampleconst_t._get_hash_recursive([]))
        return exampleconst_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", exampleconst_t._get_packed_fingerprint())[0]

