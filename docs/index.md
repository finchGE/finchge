
 
# Welcome to finchGE's documentation!

<img src="assets/images/finchgebanner.png" alt="finchGE" width="717" height="194"> 


A modern, flexible, and powerful Python framework for Grammatical Evolution.

**finchGE** is an evolutionary algorithm toolkit that evolves solutions using a grammar-based approach. Unlike traditional genetic programming, it uses a genotype-to-phenotype mapping system driven by formal grammars, enabling users to easily evolve structured programs, expressions, or models in domains such as symbolic regression, AI, and optimization.

## Why finchGE

 
- Modular and extensible: Plug-and-play mutation,  election, fitness, and search strategies.
- Designed for research and industry: Convenient and flexible API for quicker implementation.

## Usage

Using **finchGE** is straightforward.

Step 1. Define grammar in `grammar.bnf`

Step 2. Define a Fitness function ; `fitness_fn`
 
Step 3. Create `GrammaticalEvolution` instance and run

```python
    ge_ = GrammaticalEvolution(fitness_function=fitness_fn)
    fittest = ge_.find_fittest()  
```

For further details, please check. Getting Started, API documentation

##  Citation

Please use following citation to cite **finchGE** in scientific publications:

.....

.....
 


## References

- Michael O'Neill and Conor Ryan, "Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language", Kluwer Academic Publishers, 2003.
- Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. PonyGE2: Grammatical Evolution in Python. arXiv preprint, arXiv:1703.08535, 2017.