.. _contributing:

Contributing
============
🎉 First off, thanks for taking the time to contribute to torch-pme! 🎉

If you want to contribute but feel a bit lost, do not hesitate to contact us and ask
your questions! We will happily mentor you through your first contributions.

Area of contributions
---------------------
The first and best way to contribute to torch-pme is to use it and advertise it
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

The first step when developing torch-pme is to `create a fork`_ of the main
repository on github, and then clone it locally:

.. code-block:: bash

    git clone <insert/your/fork/url/here>
    cd torch-pme

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
The testsuite is implemented using the `pytest`_ framework and should be set-up
and run in an isolated virtual environment with `tox`_. All tests can be run with

.. code-block:: bash

  tox                  # all tests

If you wish to test only specific functionalities, for example:

.. code-block:: bash

  tox -e lint          # code style
  tox -e tests         # unit tests of the main library


You can also use ``tox -e format`` to use tox to do actual formatting instead of just
testing it. Also, you may want to setup your editor to automatically apply the `ruff`_
code formatter when saving your files, there are plugins to do this with `all major
editors <https://docs.astral.sh/ruff/editors/#language-server-protocol>`_.

.. _pytest: https://pytest.org
.. _tox: https://tox.readthedocs.io
.. _ruff: https://docs.astral.sh/ruff/

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

How to Perform a Release
-------------------------

1. **Prepare a Release Pull Request**

   - Create a new Pull Request (PR) with the changes you want to release.
   - Ensure that all `CI tests <https://github.com/lab-cosmo/torch-pme/actions>`_ pass.
   - Optionally, run the tests locally to double-check.

2. **Update the Version String**

   - Update the `__version__` string in ``__init__.py`` to reflect the new version, e.g.,
     `0.1.1` for a stable release or `0.1.1rc1` for a release candidate.

3. **Update the Changelog**

   - Edit the changelog located in ``docs/src/references/changelog.rst``:
     - Add a new section for the current version, summarizing the changes.
     - Leave a placeholder section titled *Unreleased* for future updates.

4. **Merge the PR and Create a Tag**

   - After the release PR is merged, create a Git tag and push it to GitHub:

     .. code-block:: bash

        git tag -a v0.1.1 -m "Release v0.1.1"
        git push origin --tags

   - For a release candidate, the tag should include an additional dash, e.g.,
     `v0.1.1-rc1`.

5. **Finalize the GitHub Release**

   - Once the PR is merged, the CI will automatically:
     - Publish the package to PyPI.
     - Create a draft release on GitHub.
   - Update the GitHub release notes by pasting the changelog for the version.
