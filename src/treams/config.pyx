"""Configuration.

This is the place to define global configuration variables. Use, e.g.,

.. code-block::

   import treams


   treams.config.POLTYPE = 'parity'
   # The rest of the code

to change the default value globally. Currently the only global configuration variable
is ``POLTYPE`` which defines the default polarization type. It can be either `helicity`
or `parity`.
"""

POLTYPE = "helicity"


cdef float SINGULARITY_REDINCGAMMA = 1e-7

def set_SINGULARITY_REDINCGAMMA(val):
   global SINGULARITY_REDINCGAMMA
   SINGULARITY_REDINCGAMMA = val