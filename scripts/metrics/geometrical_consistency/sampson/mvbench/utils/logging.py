import os

import tqdm


def pbar(iterable, **kwargs):
    """
    A wrapper around tqdm.tqdm that disables the progress bar if the disable_progress_bar flag is set.
    """
    if os.environ.get("NO_PBAR", False):
        return iterable
    return tqdm.tqdm(iterable, **kwargs)
