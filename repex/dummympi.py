"""
Add some framework to emulate MPI behavior on a single node so no MPI
implementations have to be installed.  The following example will make code
compatible with no MPI

Example
-------
    >>> try:
    ...     from mpi4py import MPI
    >>> except:
    ...     import dummympi as MPI
    >>> 
"""


class DummyMPIComm(object):
    def __init__(self):
        """Create a dummy MPI communicator for single process work."""
        self.size = 1
        self.rank = 0

    def allgather(self, sendobj=None):
        """Simulate mpicomm.allgather on one process.

        Parameters
        ----------
        
        sendobj : obj, default=None
            Object to send

        Notes
        -----
        We have not yet implemented recvobj.
        """
        return [sendobj]

    def gather(self, sendobj=None, root=0):
        """Simulate mpicomm.gather on one process.

        Parameters
        ----------
        
        sendobj : obj, default=None
            Object to send
        root : int, default=0
            Raises ValueError if not zero!
                                
        Notes
        -----
        We have not yet implemented recvobj.
        """    
        if root != 0:
            raise(ValueError("DummyMPIComm uses only a single node."))
        
        return [sendobj]


    def bcast(self, obj=None, root=0):
        """Simulate mpicomm.bcast on one process.

        Parameters
        ----------
        
        sendobj : obj, default=None
            Object to send
        root : int, default=0
            Raises ValueError if not zero!     
                    
        Notes
        -----
        We have not yet implemented recvobj.
        """
        if root != 0:
            raise(ValueError("DummyMPIComm uses only a single node."))

        return obj

    def Abort(self, errorcode=0):
        """Simulate mpicomm.Abort on one process.

        Parameters
        ----------
        
        errorcode : int, default=0
            Object to send
                    
        Notes
        -----
        We have not yet implemented recvobj.
        """        
        raise(Exception("Dummy MPI aborted!"))
    
    def Barrier(self):
        """Simulate mpicomm.Barrier on one process.

        Parameters
        ----------

        Notes
        -----
        This is a no-op, since synchronization is a non-issue in serial
        """

    def barrier(self):
        """Same as Barrier (above)"""

# Define a COMM_WORLD
COMM_WORLD = DummyMPIComm()
