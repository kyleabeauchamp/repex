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
    """A class with timing functions.
    
    Notes
    -----
    To use this, first initialize a Timer, then call 
    timestamp(keyword) at various points in the code to be benchmarked.
    Finally, print the timings with report_timings()
    """
    
    def __init__(self):
        self.reset_timing_statistics()


    def reset_timing_statistics(self):
        """Reset the timing statistics.
        """

        self.timestamps = OrderedDict()
        
        self._t0 = {}
        self._t1 = {}
        self._elapsed = {}

    def check_initialized(self):
        for key in ["timestamps", "_t0", "_t1", "_elapsed"]:
            if not hasattr(self, key):
                self.reset_timing_statistics()
            
    
    def timestamp(self, keyword):
        """Record the current time and save as keyword.
        
        Parameters
        ----------
        keyword : str
            What name to associate the current time with.
        """


        self.check_initialized()
        self.timestamps[keyword] = time.time()
    
    def start(self, keyword):
        """Start a timer with given keyword."""
        
        self.check_initialized()        
        self._t0[keyword] = time.time()
    
    def stop(self, keyword):
        if keyword in self._t0:
            self._t1[keyword] = time.time()
            self._elapsed[keyword] = self._t1[keyword] - self._t0[keyword]
        else:
            logger.info("Can't stop timing for keyword")

    def _report_timestamps(self):
        
        logger.debug("Saved timestamp differences:")
        
        for i, keyword in enumerate(self.timestamps):
            try:
                keyword2 = self.timestamps.keys()[i + 1]
                delta = self.timestamps[keyword2] - self.timestamps[keyword]
                logger.debug("%24s %8.3f s" % (keyword, delta))                
            except IndexError:
                pass
            

    def _report_stopwatch(self):
        
        logger.debug("Saved stopwatch times:")

        for keyword, time in self._elapsed.iteritems():
            logger.debug("%24s %8.3f s" % (keyword, time))


    def report_timing(self, clear=True):
        """
        Report the timing for a move type.
        
        """
        self.check_initialized()

        self._report_timestamps()
        self._report_stopwatch()
        
        if clear == True:
            self.reset_timing_statistics()
        
        return
