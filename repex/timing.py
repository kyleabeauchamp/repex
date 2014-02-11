import time
from functools import wraps
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)


TIMINGS = {}  # Global dictionary of stored timings, for debugging purposes


def benchmark(function, name=None):
    """A decorator for timing functions."""
    if name is not None:
        name = function.__name__

    @wraps(function)
    def timed(*args, **kw):
        start = time.time()
        result = function(*args, **kw)
        end = time.time()
        
        delta = end - start
            
        TIMINGS[name] = delta
        logger.debug("Benchmarking %s: Start: %f.  End: %f.  Delta: %f" % (name, start, end, delta))
        return result
    return timed

class TimeContext(object):
    """A timing context manager

    Example
    -------
    >>> long_function = lambda : None
    >>> with TimeContext('long_function'):
    ...     long_function()
    """
    def __init__(self, name='block'):
        self.name = name
        self.time = 0
        self.start = None
        self.end = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, ty, val, tb):
        self.end = time.time()
        self.time = self.end - self.start
        logger.debug("%s: %0.3f seconds" % (self.name, self.time))
        TIMINGS[self.name] = self.time
        return False


class Timer(object):
    """A Mixin class with timing functions."""
    
    def __init__(self):
        self.reset_timing_statistics()


    def reset_timing_statistics(self):
        """Reset the timing statistics.
        """

        self.elapsed_time = OrderedDict()
        self.timestamps = OrderedDict()
        self.timestamps["reset_timing_statistics"] = time.time()

    
    def timestamp(self, keyword):
        """Record the current time and save as keyword.
        
        Parameters
        ----------
        keyword : str
            What name to associate the current time with.
        """

        if not hasattr(self, "elapsed_time"):
            self.reset_timing_statistics()
        
        self.timestamps[keyword] = time.time()
    
    def record_timing(self, timing_keyword, elapsed_time):
        """
        Record the elapsed time for a given phase of the calculation.

        Parameters
        ----------
        timing_keyword : str
           The keyword for which to store the elapsed time (e.g. 'context creation', 'integration', 'state extraction')
        elapsed_time : float
           The elapsed time, in seconds.

        """
        if not hasattr(self, "elapsed_time"):
            self.reset_timing_statistics()
        
        if timing_keyword not in self.elapsed_time:
            self.elapsed_time[timing_keyword] = 0.0
        self.elapsed_time[timing_keyword] += elapsed_time

    def report_timing(self, clear=True):
        """
        Report the timing for a move type.
        
        """
        if not hasattr(self, "elapsed_time"):
            self.reset_timing_statistics()

        self.timestamp("At report")
        
        logger.debug("Saved elapsed times:")
        for timing_keyword in self.elapsed_time:
            logger.debug("%24s %8.3f s" % (timing_keyword, self.elapsed_time[timing_keyword]))

        logger.debug("Saved timestamp differences:")
        
        for i, keyword in enumerate(self.timestamps):
            
            try:
                keyword2 = self.timestamps.keys()[i + 1]
            except IndexError:
                pass
            delta = self.timestamps[keyword2] - self.timestamps[keyword]
            logger.debug("%24s %8.3f s" % (keyword, delta))

        if clear == True:
            self.reset_timing_statistics()
        
        return
