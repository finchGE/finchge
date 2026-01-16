# FinchGE: A Modular Grammatical Evolution Library
 
[![PyPI](https://img.shields.io/pypi/v/finchge?color=blue&label=PyPI)](https://pypi.org/project/finchge/)
[![Python](https://img.shields.io/pypi/pyversions/finchge?color=blue)](https://pypi.org/project/finchge/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/readthedocs/finchge?color=blue)](https://finchge.readthedocs.io/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/finchGE/finchge)

FinchGE is a modern, modular, and user-friendly Python library for Grammatical Evolution (GE) - a powerful evolutionary algorithm that uses formal grammars to evolve programs, expressions, and solutions.

## Features

- Define grammars using BNF-style syntax
- Supports standard genetic operations: mutation, crossover, selection
- Flexible fitness evaluation for any problem domain
- Modular and extensible design allowing conveniently plugin custom Algorithms and Operators
- Easy-to-read in-built logging and visualization 
- Intuitive API with extensive documentation and examples

## Installation

```bash
# Basic installation
pip install finchge

# With optional dependencies 
pip install finchge[pytorch]    # PyTorch support for using pytorch models (for HPO or NAS)
pip install finchge[all]        # All optional dependencies
```
 
 ## Documentation

Comprehensive documentation is available at [finchge.readthedocs.io](https://finchge.readthedocs.io/) including:

- [Tutorials](https://finchge.readthedocs.io/en/latest/tutorials) - Step-by-step guides
- [Examples](https://finchge.readthedocs.io/en/latest/examples) - Real-world use cases
- [API Reference](https://finchge.readthedocs.io/en/latest/api/) - Complete API documentation
 
## Architecture

FinchGE is built with modularity in mind:
 

## Contributing

All contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Bug Reports and Feature Requests

Found a bug or have a feature request? Please [open an issue](https://github.com/finchge/finchge/issues) on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Third-Party Dependencies

FinchGE includes the following third-party software:

| Package | License |
|---------|---------|
| tqdm | MPLv2.0, MIT |
| diskcache | Apache 2.0 |
| plotly | MIT |
| tabulate | MIT |
| matplotlib | Python Software Foundation License |
| pandas | BSD 3-Clause | 
| scikit-learn | BSD 3-Clause |

All packages retain their original licenses.


## Alpha Release Notice
Note: This is version ```0.1.0a1``` - an alpha release. Expect breaking changes and bugs. Not production ready.

What to expect:
- Bugs and unexpected behavior
- Rapid API changes
- Frequent updates
- Limited test coverage (improving daily)

 
## Acknowledgments

- Inspired by PonyGE2
- Built with the Python scientific ecosystem (numpy, scikit-learn, matplotlib)