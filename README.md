# M-Brain

This repository contains experiments exploring computational
implementations inspired by the Thousand Brains Theory of the neocortex.

The code in `src/` provides a minimal simulation consisting of
cortical columns that learn feature/location pairs via path integration
and reach a consensus through voting.

## Running the demo

The demonstration script can be executed with:

```bash
python -m src.simulation
```

It builds an object model with simple geometric shapes and shows how a
collection of columns can vote on the object identity from partial
features.

