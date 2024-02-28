MeshLODE
========

|tests| |codecov| |docs|

Particle-mesh based calculation of Long Distance Equivariants.

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://meshlode.readthedocs.io

.. marker-installation

Installation
------------

You can install *MeshLode* using pip with

.. code-block:: bash

    git clone https://github.com/lab-cosmo/MeshLODE
    cd MeshLODE
    pip install .

You can then ``import meshlode`` and use it in your projects!

We also provide bindings to `metatensor
<https://lab-cosmo.github.io/metatensor/latest/>`_ which can optionally be installed
together and used as ``meshlode.metatensor`` via

.. code-block:: bash

    pip install .[metatensor]

.. marker-issues

Having problems or ideas?
-------------------------

Having a problem with MeshLODE? Please let us know by `submitting an issue
<https://github.com/lab-cosmo/MeshLODE/issues>`_.

Submit new features or bug fixes through a `pull request
<https://github.com/lab-cosmo/MeshLODE/pulls>`_.

.. marker-contributing

Contributors
------------

Thanks goes to all people that make MeshLODE possible:

.. image:: https://contrib.rocks/image?repo=lab-cosmo/MeshLODE
   :target: https://github.com/lab-cosmo/MeshLODE/graphs/contributors

.. |tests| image:: https://github.com/lab-cosmo/MeshLODE/workflows/Test/badge.svg
   :alt: Github Actions Tests Job Status
   :target: (https://github.com/lab-cosmo/MeshLODE/\
                actions?query=workflow%3ATests)

.. |codecov| image:: https://codecov.io/gh/lab-cosmo/meshlode/branch/main/graph/badge.svg?token=UZJPJG34SM
   :alt: Code coverage
   :target: https://codecov.io/gh/lab-cosmo/meshlode/

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Python
   :target: https://meshlode.readthedocs.io
