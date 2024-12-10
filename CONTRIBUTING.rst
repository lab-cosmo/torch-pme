.. _contributing:

Contributing
============
🎉 First off, thanks for taking the time to contribute to metatrain! 🎉

If you want to contribute but feel a bit lost, do not hesitate to contact us and ask
your questions! We will happily mentor you through your first contributions.

Area of contributions
---------------------
The first and best way to contribute to metatrain is to use it and advertise it
to other potential users. Other than that, you can help with:

- documentation: correcting typos, making various documentation clearer;
- bug fixes and improvements to existing code;
- adding new architectures
- and many more ...

All these contributions are very welcome. We accept contributions via Github `pull
request <https://github.com/lab-cosmo/torch-pme/issues>`_. If you want to work on the code
and pick something easy to get started, have a look at the `good first issues
<https://github.com/lab-cosmo/torch-pme/labels/Good%20first%20issue>`_.

Required tools
--------------

You will need to install and get familiar with the following tools when working
on torch-pme:

- **git**: the software we use for version control of the source code. See
  https://git-scm.com/downloads for installation instructions.
- **Python**: you can install ``Python`` and ``pip`` from your operating system.
  We require a Python version of at least 3.9.
- **tox**: a Python test runner, cf https://tox.readthedocs.io/en/latest/. You
  can install tox with ``pip install tox``.

Getting the code
----------------

The first step when developing metatensor is to `create a fork`_ of the main
repository on github, and then clone it locally:

.. code-block:: bash

    git clone <insert/your/fork/url/here>
    cd metatensor

    # setup the local repository so that the main branch tracks changes in
    # the original repository
    git remote add upstream https://github.com/lab-cosmo/torch-pme.git
    git fetch upstream
    git branch main --set-upstream-to=upstream/main

Once you get the code locally, you will want to run the tests to check
everything is working as intended. See the next section on this subject.

If everything is working, you can create your own branches to work on your
changes:

.. code-block:: bash

    git checkout -b <my-branch-name>
    # code code code

    # push your branch to your fork
    git push -u origin <my-branch-name>
    # follow the link in the message to open a pull request (PR)

.. _create a fork: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo

Running the tests
-----------------
The testsuite is implemented using Python's `unittest`_ framework and should be set-up
and run in an isolated virtual environment with `tox`_. All tests can be run with

.. code-block:: bash

  tox                  # all tests

If you wish to test only specific functionalities, for example:

.. code-block:: bash

  tox -e lint          # code style
  tox -e tests         # unit tests of the main library


You can also use ``tox -e format`` to use tox to do actual formatting instead of just
testing it. Also, you may want to setup your editor to automatically apply the `black
<https://black.readthedocs.io/en/stable/>`_ code formatter when saving your files, there
are plugins to do this with `all major editors
<https://black.readthedocs.io/en/stable/editor_integration.html>`_.

.. _unittest: https://docs.python.org/3/library/unittest.html
.. _tox: https://tox.readthedocs.io/en/latest

Contributing to the documentation
---------------------------------
The documentation is written in reStructuredText (rst) and uses `sphinx`_ documentation
generator. In order to modify the documentation, first create a local version on your
machine as described above. Then, build the documentation with

.. code-block:: bash

    tox -e docs

You can then visualize the local documentation with your favorite browser using the
following command (or open the :file:`docs/build/html/index.html` file manually).

.. code-block:: bash

    # on linux, depending on what package you have installed:
    xdg-open docs/build/html/index.html
    firefox docs/build/html/index.html

    # on macOS:
    open docs/build/html/index.html

.. _`sphinx` : https://www.sphinx-doc.org
