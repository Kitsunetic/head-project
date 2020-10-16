import multiprocessing
from typing import Any, Callable, Iterable

from tqdm import tqdm


def xqdm(func: Callable, iterable: Iterable[Any], num_workers: int = 0, **kargs):
    """
    func  : Callable
        Callable to run with a progressbar.
    iterable  : iterable, optional
        Iterable to decorate with a progressbar.
        Leave blank to manually manage the updates.
    num_workers: int, optional
        The number of processes to create.
        If `None` or lesser than 1, it will create processes as much as `multiprocessing.cpu_count()`.
    desc  : str, optional
        Prefix for the progressbar.
    leave  : bool, optional
        If [default: True], keeps all traces of the progressbar
        upon termination of iteration.
        If `None`, will leave only if `position` is `0`.
    file  : `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    ncols  : int, optional
        The width of the entire output message. If specified,
        dynamically resizes the progressbar to stay within this bound.
        If unspecified, attempts to use environment width. The
        fallback is a meter width of 10 and no limit for the counter and
        statistics. If 0, will not print any meter (only stats).
    mininterval  : float, optional
        Minimum progress display update interval [default: 0.1] seconds.
    maxinterval  : float, optional
        Maximum progress display update interval [default: 10] seconds.
        Automatically adjusts `miniters` to correspond to `mininterval`
        after long display update lag. Only works if `dynamic_miniters`
        or monitor thread is enabled.
    miniters  : int or float, optional
        Minimum progress display update interval, in iterations.
        If 0 and `dynamic_miniters`, will automatically adjust to equal
        `mininterval` (more CPU efficient, good for tight loops).
        If > 0, will skip display of specified number of iterations.
        Tweak this and `mininterval` to get very efficient loops.
        If your progress is erratic with both fast and slow iterations
        (network, skipping items, etc) you should set miniters=1.
    ascii  : bool or str, optional
        If unspecified or False, use unicode (smooth blocks) to fill
        the meter. The fallback is to use ASCII characters " 123456789#".
    disable  : bool, optional
        Whether to disable the entire progressbar wrapper
        [default: False]. If set to None, disable on non-TTY.
    unit  : str, optional
        String that will be used to define the unit of each iteration
        [default: it].
    unit_scale  : bool or int or float, optional
        If 1 or True, the number of iterations will be reduced/scaled
        automatically and a metric prefix following the
        International System of Units standard will be added
        (kilo, mega, etc.) [default: False]. If any other non-zero
        number, will scale `total` and `n`.
    dynamic_ncols  : bool, optional
        If set, constantly alters `ncols` to the environment (allowing
        for window resizes) [default: False].
    smoothing  : float, optional
        Exponential moving average smoothing factor for speed estimates
        (ignored in GUI mode). Ranges from 0 (average speed) to 1
        (current/instantaneous speed) [default: 0.3].
    bar_format  : str, optional
        Specify a custom bar string formatting. May impact performance.
        [default: '{l_bar}{bar}{r_bar}'], where
        l_bar='{desc}: {percentage:3.0f}%|' and
        r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
          '{rate_fmt}{postfix}]'
        Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
          percentage, elapsed, elapsed_s, ncols, desc, unit,
          rate, rate_fmt, rate_noinv, rate_noinv_fmt,
          rate_inv, rate_inv_fmt, postfix, unit_divisor,
          remaining, remaining_s.
        Note that a trailing ": " is automatically removed after {desc}
        if the latter is empty.
    initial  : int or float, optional
        The initial counter value. Useful when restarting a progress
        bar [default: 0]. If using float, consider specifying `{n:.3f}`
        or similar in `bar_format`, or specifying `unit_scale`.
    position  : int, optional
        Specify the line offset to print this bar (starting from 0)
        Automatic if unspecified.
        Useful to manage multiple bars at once (eg, from threads).
    postfix  : dict or *, optional
        Specify additional stats to display at the end of the bar.
        Calls `set_postfix(**postfix)` if possible (dict).
    unit_divisor  : float, optional
        [default: 1000], ignored unless `unit_scale` is True.
    write_bytes  : bool, optional
        If (default: None) and `file` is unspecified,
        bytes will be written in Python 2. If `True` will also write
        bytes. In all other cases will default to unicode.
    lock_args  : tuple, optional
        Passed to `refresh` for intermediate output
        (initialisation, iterating, and updating).
    gui  : bool, optional
        WARNING: internal parameter - do not use.
        Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
        matplotlib animations for a graphical output [default: False].
    """
    iterable = list(iterable)
    length = len(iterable)

    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        results = [None for _ in range(length)]
        with tqdm(total=length, **kargs) as t:
            for i, v in enumerate(pool.imap_unordered(func, iterable)):
                results[i] = v
                t.update()

    return results
