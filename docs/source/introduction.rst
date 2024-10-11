Introduction to SGMethods
==========================
SGMethods is a Python object-oriented code for sparse grid interpolation (also
called stochastic collocation) with a focus on parametric coefficient PDEs.

Features
---------------------
* Several nodes, multi-index sets, and interpolation methods are implemented...
* Or you can define and use your own following the examples
* Single- and multilevel methods
* A-priori and adaptive sparse grid construction (Coming soon)
* Tests wit pytest
* Documentation in sphinx

Installation
------------
For the moment, only the repository is available. In the future a package will
be released on PyPI.

1. Clone the repository::

    git clone git@github.com:andreascaglioni/SGMethods.git

or::

    git clone https://github.com/andreascaglioni/SGMethods.git

2. Install the dependencies::

    pip install -r requirements.txt

3. Run the tests from the root directory::

    pytest tests/test_*.py

Usage
-----
See the examples in the omonymous directory.

Documentation for all the modules is also provided below.

Contributing
------------
Contributions are welcome! Please open an issue or submit a pull request.

License
-------
Distributed under the MIT License. See LICENSE for more information.

Contact
-------
Andrea Scaglioni - 
`Get in touch on my website <https://andreascaglioni.net/contacts>`


From
`README.md on GitHub <https://github.com/andreascaglioni/SGMethods>`_ .