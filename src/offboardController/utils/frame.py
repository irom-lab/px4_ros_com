import numpy as np


# == frame stacking and skipping ==
def get_frames(prev_obs, traj_size, frame_skip):
    """
    Assume at least one obs in prev_obs - reasonable since we can always get the initial obs, and then get frames.
    """
    traj_cover = (traj_size-1) * frame_skip + traj_size
    default_frame_seq = np.arange(0, traj_cover, (frame_skip + 1))

    if len(prev_obs) == 1:
        seq = np.zeros((traj_size), dtype='int')
    elif len(prev_obs) < traj_cover:  # always pick zero (most recent one)
        seq_random = np.random.choice(
            np.arange(1, len(prev_obs)), traj_size - 1, replace=True
        )
        seq_random = np.sort(seq_random)  # ascending
        seq = np.insert(seq_random, 0, 0)  # add to front, then flipped
    else:
        seq = default_frame_seq
    seq = np.flip(seq)  #! since prev_obs appends left
    obs_stack = np.concatenate([prev_obs[obs_ind] for obs_ind in seq])
    return obs_stack
