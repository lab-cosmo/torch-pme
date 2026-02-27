:orphan:

.. _userdoc-how-to:

Examples
========

This section lists introductory examples and recipes to the various classes and
functions of ``torch-pme``. Example are sorted in increasing order of complexity. For
details on the API specification of the functions take a look at the
:ref:`userdoc-reference` section.

To run the all examples, install ``torchpme`` with the ``examples`` and ``metatensor``
optional dependencies.

.. code-block:: bash

    pip install .[examples,metatensor]



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In a physical system, the (electrical) charge is a scalar atomic property, and besides the distance between the particles, the charge defines the electrostatic potential. When computing a potential with Meshlode, you can not only pass a (reshaped) 1-D array containing the charges to the compute method of calculators, but you can also pass a 2-D array containing multiple charges per atom. Meshlode will then calculate one potential per so-called charge-channel. For the standard electrostatic potential, the number of charge channels is 1. Additional charge channels are especially useful in a machine learning task where, for example, one wants to create one potential for each species in an atomistic system using a so-called one-hot encoding of charges.">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_1-charges-example_thumb.png
    :alt:

  :ref:`sphx_glr_examples_1-charges-example.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Computations with Multiple Charge Channels</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Accurately calculating forces as derivatives from energy is crucial for predicting system dynamics as well as in training machine learning models. In systems where forces are derived from the gradients of the potential energy, it is essential that the distance calculations between particles are included in the computational graph. This ensures that the force computations respect the dependencies between particle positions and distances, allowing for accurate gradients during backpropagation.">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_2-neighbor-lists-usage_thumb.png
    :alt:

  :ref:`sphx_glr_examples_2-neighbor-lists-usage.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Advanced neighbor list usage</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=":Authors: Michele Ceriotti @ceriottm">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_3-mesh-demo_thumb.png
    :alt:

  :ref:`sphx_glr_examples_3-mesh-demo.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples of the MeshInterpolator class</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=":Authors: Michele Ceriotti @ceriottm">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_4-kspace-demo_thumb.png
    :alt:

  :ref:`sphx_glr_examples_4-kspace-demo.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples of the KSpaceFilter class</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=":Authors: Michele Ceriotti @ceriottm">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_5-autograd-demo_thumb.png
    :alt:

  :ref:`sphx_glr_examples_5-autograd-demo.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom models with automatic differentiation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=":Authors: Michele Ceriotti @ceriottm">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_6-splined-potential_thumb.png
    :alt:

  :ref:`sphx_glr_examples_6-splined-potential.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Splined potentials</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=":Authors: Michele Ceriotti @ceriottm">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_7-lode-demo_thumb.png
    :alt:

  :ref:`sphx_glr_examples_7-lode-demo.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Computing LODE descriptors</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=":Authors: Egor Rumiantsev @E-Rum;    Philip Loche @PicoCentauri">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_8-combined-potential_thumb.png
    :alt:

  :ref:`sphx_glr_examples_8-combined-potential.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optimizing a linear combination of potentials</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we demonstrate how to construct a metatensor atomistic model  based on the metatensor interface &lt;metatensor&gt; of torchpme. The model will be used to run a very short molecular dynamics (MD) simulation of a non-neutral hydroden plasma in a cubic box. The plasma consists of massive point particles which are interacting pairwise via a Coulomb force which we compute using the EwaldCalculator.">

.. only:: html

  .. image:: /examples/images/thumb/sphx_glr_9-atomistic-model_thumb.png
    :alt:

  :ref:`sphx_glr_examples_9-atomistic-model.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Atomistic model for molecular dynamics</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/1-charges-example
   /examples/2-neighbor-lists-usage
   /examples/3-mesh-demo
   /examples/4-kspace-demo
   /examples/5-autograd-demo
   /examples/6-splined-potential
   /examples/7-lode-demo
   /examples/8-combined-potential
   /examples/9-atomistic-model


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: examples_python.zip </examples/examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: examples_jupyter.zip </examples/examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
