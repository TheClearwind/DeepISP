import numpy as np


def extract_bayer_channels(raw):
    # RGGB模式
    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    return ch_R, ch_Gr, ch_B, ch_Gb


def extract_bayer_channels_v2(raw):
    # RGGB模式
    ch_R = np.zeros_like(raw)
    ch_G = np.zeros_like(raw)
    ch_B = np.zeros_like(raw)
    ch_B[1::2, 1::2] = raw[1::2, 1::2]
    ch_R[0::2, 0::2] = raw[0::2, 0::2]
    ch_G[0::2, 1::2] = raw[0::2, 1::2]
    ch_G[1::2, 0::2] = raw[1::2, 0::2]

    return ch_R, ch_G, ch_B


def pack_raw(raw):
    ch_B, ch_Gb, ch_R, ch_Gr = extract_bayer_channels(raw)
    packed = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    packed = np.transpose(packed, [2, 0, 1])
    return packed


def pack_raw_v2(raw):
    ch_R, ch_G, ch_B = extract_bayer_channels_v2(raw)
    packed = np.dstack((ch_R, ch_G, ch_B))
    packed = np.transpose(packed, [2, 0, 1])
    return packed


if __name__ == '__main__':
    raw = np.random.randn(224, 224)
    packed = pack_raw_v2(raw)
    print(packed.shape)
