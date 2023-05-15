Contributor's Guide
===================

Installation
---------------
Clone the repository and install the requirements, which are needed to develop.

..  code-block:: bash

    make dev-install


Test the installation by running tests and lints:

..  code-block:: bash

    make test # This will run all tests
    make lint # This will lint the simba_ml source code
    make tests-lint # This will lint the tests

    # or

    make check # This will run all tests and lints
 

Coding Standards
----------------
- Code must succeed the CI-Pipeline using pylint and pycodestyle
  You can test this locally by running

  .. code-block:: bash

    make check

- Code should be tested and be fully covered by the tests

- Docstrings should be written according to the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_.