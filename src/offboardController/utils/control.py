import numpy as np


def quat2Dcm(q):
    dcm = np.zeros([3, 3])

    dcm[0, 0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    dcm[0, 1] = 2.0 * (q[1] * q[2] - q[0] * q[3])
    dcm[0, 2] = 2.0 * (q[1] * q[3] + q[0] * q[2])
    dcm[1, 0] = 2.0 * (q[1] * q[2] + q[0] * q[3])
    dcm[1, 1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    dcm[1, 2] = 2.0 * (q[2] * q[3] - q[0] * q[1])
    dcm[2, 0] = 2.0 * (q[1] * q[3] - q[0] * q[2])
    dcm[2, 1] = 2.0 * (q[2] * q[3] + q[0] * q[1])
    dcm[2, 2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    return dcm


def RotToQuat(R):

    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]
    # From page 68 of MotionGenesis book
    tr = R11 + R22 + R33

    if tr > R11 and tr > R22 and tr > R33:
        e0 = 0.5 * np.sqrt(1 + tr)
        r = 0.25 / e0
        e1 = (R32 - R23) * r
        e2 = (R13 - R31) * r
        e3 = (R21 - R12) * r
    elif R11 > R22 and R11 > R33:
        e1 = 0.5 * np.sqrt(1 - tr + 2 * R11)
        r = 0.25 / e1
        e0 = (R32 - R23) * r
        e2 = (R12 + R21) * r
        e3 = (R13 + R31) * r
    elif R22 > R33:
        e2 = 0.5 * np.sqrt(1 - tr + 2 * R22)
        r = 0.25 / e2
        e0 = (R13 - R31) * r
        e1 = (R12 + R21) * r
        e3 = (R23 + R32) * r
    else:
        e3 = 0.5 * np.sqrt(1 - tr + 2 * R33)
        r = 0.25 / e3
        e0 = (R21 - R12) * r
        e1 = (R13 + R31) * r
        e2 = (R23 + R32) * r

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = np.array([e0, e1, e2, e3])
    q = q * np.sign(e0)

    q = q / np.sqrt(np.sum(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2))

    return q


# Quaternion multiplication
def quatMultiply(q, p):
    Q = np.array([[q[0], -q[1], -q[2], -q[3]], [q[1], q[0], -q[3], q[2]],
                  [q[2], q[3], q[0], -q[1]], [q[3], -q[2], q[1], q[0]]])
    return Q @ p


# Inverse quaternion
def quatInverse(q):
    qinv = np.array([q[0], -q[1], -q[2], -q[3]]) / np.linalg.norm(q)
    return qinv
