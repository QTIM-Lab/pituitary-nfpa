# `pituitary-nfpa`

Project for predicting recurrence in non-functioning pituitary adenomas.
This source code is released to accompany the following paper:

- Machine Learning-Based Prediction Of Tumor Recurrence Or Progression In
Patients With Non-Functioning Pituitary Adenomas. Ukasha Habib, Christopher P
Bridge, Masaaki Mikamoto, Otto Rapalino, Albert E Kim, Karen K Miller, Brooke
Swearingen, Pamela S Jones, and Nicholas A Tritos

It is released "as-is" for the purposes of scientific transparency only, is not
intended to be re-usable code, and we have no plans to maintain or improve it.

### Installation

The codebase is configured as a Python package that you can install on your
system (along with all its dependencies) by running this command from the root
of the project directory:

```bash
$ pip install -e .
```

### CLI Usage

The Python package provides a command line interface (CLI) when executed using
the `python -m pituitary_nfpa` command, which should be
available whenever the environment that the package is on findable (either
because it is installed or added to the `PYTHONPATH`). There are various
sub-commands, each for a different process associated with the project, e.g.
training, testing, preprocessing. Each sub-command corresponds to a Python
module in the `pituitary_nfpa.commands` package.

To see a list of available sub-commands, use:

```bash
python -m pituitary_nfpa --help
```

To see the documentation of an individual command, e.g. `train` use:

```bash
python -m pituitary_nfpa train --help
```
