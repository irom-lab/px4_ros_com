"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class flowdrone_t(object):
    __slots__ = ["timestamp", "drone_state", "thrust_sp", "thrust_residual", "body_rate_sp", "body_rate_residual", "wind_magnitude_estimate", "wind_obs_current", "wind_angle_estimate"]

    __typenames__ = ["int64_t", "double", "double", "double", "double", "double", "double", "double", "double"]

    __dimensions__ = [None, [20], None, None, [3], [3], None, [5], None]

    def __init__(self):
        self.timestamp = 0
        self.drone_state = [ 0.0 for dim0 in range(20) ]
        self.thrust_sp = 0.0
        self.thrust_residual = 0.0
        self.body_rate_sp = [ 0.0 for dim0 in range(3) ]
        self.body_rate_residual = [ 0.0 for dim0 in range(3) ]
        self.wind_magnitude_estimate = 0.0
        self.wind_obs_current = [ 0.0 for dim0 in range(5) ]
        self.wind_angle_estimate = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(flowdrone_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">q", self.timestamp))
        buf.write(struct.pack('>20d', *self.drone_state[:20]))
        buf.write(struct.pack(">dd", self.thrust_sp, self.thrust_residual))
        buf.write(struct.pack('>3d', *self.body_rate_sp[:3]))
        buf.write(struct.pack('>3d', *self.body_rate_residual[:3]))
        buf.write(struct.pack(">d", self.wind_magnitude_estimate))
        buf.write(struct.pack('>5d', *self.wind_obs_current[:5]))
        buf.write(struct.pack(">d", self.wind_angle_estimate))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != flowdrone_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return flowdrone_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = flowdrone_t()
        self.timestamp = struct.unpack(">q", buf.read(8))[0]
        self.drone_state = struct.unpack('>20d', buf.read(160))
        self.thrust_sp, self.thrust_residual = struct.unpack(">dd", buf.read(16))
        self.body_rate_sp = struct.unpack('>3d', buf.read(24))
        self.body_rate_residual = struct.unpack('>3d', buf.read(24))
        self.wind_magnitude_estimate = struct.unpack(">d", buf.read(8))[0]
        self.wind_obs_current = struct.unpack('>5d', buf.read(40))
        self.wind_angle_estimate = struct.unpack(">d", buf.read(8))[0]
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if flowdrone_t in parents: return 0
        tmphash = (0x34f1a9893c8cc9f0) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if flowdrone_t._packed_fingerprint is None:
            flowdrone_t._packed_fingerprint = struct.pack(">Q", flowdrone_t._get_hash_recursive([]))
        return flowdrone_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", flowdrone_t._get_packed_fingerprint())[0]

