~~~~~~~~~~~~~~
with ``mamba``
~~~~~~~~~~~~~~


``spudtr`` is packaged for installation with ``mamba`` or
``conda``. The following examples show how to install into a fresh
conda environment with some useful various options.


On computers with high performance Intel CPUs you may get better
performance with MKL accelerated libraries.

.. code-block:: bash

   mamba create --name spudtr_env -c conda-forge spudtr "blas=*=mkl*"


On AMD CPUs OpenBLAS may be faster.

.. code-block:: bash

   mamba create --name spudtr_env -c conda-forge spudtr "blas=*=openblas*"


If you need to a specific version of Python or other packages use
``conda`` package selection syntax.

.. code-block:: bash

   mamba create --name spudtr_env -c conda-forge spudtr "blas=*=openblas*" python=3.8 jupyter-lab


~~~~~~~~~~~~~~
with ``conda``
~~~~~~~~~~~~~~


If ``mamba`` is not available, you can use ``conda``.
		

.. code-block:: bash

   conda create --name spudtr_env -c conda-forge spudtr "blas=*=mkl*"
