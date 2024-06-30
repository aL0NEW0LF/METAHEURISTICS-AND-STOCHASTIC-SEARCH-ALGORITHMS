# **METAHEURISTICS AND STOCHASTIC SEARCH ALGORITHMS**

A collection of metaheuristics and stochastic search algorithms implemented in Python. This may or may not be developed into a package in the future, or contributed to an existing package.

## **Algorithms**

- [x] Ali Baba and the Forty Thieves
- [x] Virus Colony Search
- [x] Coronavirus herd immunity optimization

## **Algorithm Details**
### **Ali Baba and the Forty Thieves**
Ali Baba and the Forty Thieves is a novel meta-heuristic algorithm for solving numerical optimization problems. The algorithm is inspired by the story of Ali Baba and the Forty Thieves. The algorithm is based on the idea of a group of thieves that work together to find the best solution to a problem. The algorithm is designed to be simple and easy to implement, and it has been shown to be effective in solving a wide range of optimization problems.

**Algorithm Steps**
1. Initialize the population of thieves.
2. Evaluate the fitness of each thief.
3. Select the best thief as the leader.
4. Generate a new population of thieves by applying the following operations:
    - Stealing: Each thief steals a random amount of treasure from the leader.
    - Hiding: Each thief hides a random amount of treasure in a random location.
    - Sharing: Each thief shares a random amount of treasure with a random thief.
5. Evaluate the fitness of each thief in the new population.
6. Select the best thief as the leader.
7. Repeat steps 4-6 until a stopping criterion is met.

**Parameters**
- `noThieves`: The number of thieves in the population.
- `lb`: The lower bound of the search space.
- `ub`: The upper bound of the search space.
- `dim`: The dimension of the search space.
- `fobj`: The objective function to be optimized.
- `itemax` (optional): The maximum number of iterations.

## **Usage**
To start off, clone this branch of the repo into your local:

```shell
git clone https://github.com/aL0NEW0LF/METAHEURISTICS-AND-STOCHASTIC-SEARCH-ALGORITHMS.git
```

Then, navigate to the directory where you cloned the repo:

```shell
cd METAHEURISTICS-AND-STOCHASTIC-SEARCH-ALGORITHMS
```

Create a virtual environment:

**Windows**

```shell
py -3 -m venv .venv
```

**MacOS/Linus**

```shell
python3 -m venv .venv
```

Then, activate the env:

**Windows**

```shell
.venv\Scripts\activate
```

**MacOS/Linus**

```shell
. .venv/bin/activate
```

You can run the following command to install the dependencies:

```shell
pip3 install -r requirements.txt
```

The main file is used to run the algorithms. You can run the following command to run the main file:

```shell
python3 main.py
```

## **Cite me**
**BiBTeX**
```bibtex
@misc{aL0NEW0LF_Metaheuristics, 
title={Metaheuristics and stochastic search algorithms implementation}, 
url={https://github.com/aL0NEW0LF/METAHEURISTICS-AND-STOCHASTIC-SEARCH-ALGORITHMS}, 
journal={GitHub}, 
author={Fakhre-Eddine, Mohamed Amine}, 
year={2024}} 
```

**APA7**
```
Fakhre-Eddine, M. A. (2024). Metaheuristics and stochastic search algorithms implementation. GitHub. https://github.com/aL0NEW0LF/METAHEURISTICS-AND-STOCHASTIC-SEARCH-ALGORITHMS 
```

## **References**
- Braik, M., Ryalat, M.H. & Al-Zoubi, H. A novel meta-heuristic algorithm for solving numerical optimization problems: Ali Baba and the forty thieves. Neural Comput & Applic 34, 409–455 (2022). https://doi.org/10.1007/s00521-021-06392-x

- Mu Dong Li, Hui Zhao, Xing Wei Weng, & Tong Han (2016). A novel nature-inspired algorithm for optimization: Virus colony search. Advances in Engineering Software, 92, 65-88. https://doi.org/10.1016/j.advengsoft.2015.11.004

- Al-Betar, M.A., Alyasseri, Z.A.A., Awadallah, M.A.et al.Coronavirus herdimmunityoptimizer(CHIO).Neural Comput & Applic33, 5011–5042 (2021). https://doi.org/10.1007/s00521-020-05296-6

- Samady, A. Samashi47/metaheuristics: Implementation of various metaheuristic algorithms in C++ and python. GitHub. https://github.com/Samashi47/Metaheuristics