
# How To Contribute

Thank you for considering contributing to `eurocropsml`!
Everyone is very welcome to help improve it.

This document is intended to help you get started and make the process of contributing more accessible. Do not be afraid ask if something is unclear!


## Workflow

- Every contribution is welcome, no matter how small!
  Do not hesitate to submit fixes for typos etc.
- Try to stick to only *one* change or fix per merge/pull request.
- Add tests and documentation for your code.
  Contributions missing tests or documentation can not be merged.
- Make sure all changes pass our CI.
  We will usually not give any feedback until the CI is green unless you ask explicitly for it.
- Once you have addressed review feedback bump the pull request with a short note, so we know you are done.


### Branching Strategy

We use a simple feature branches approach.

- Use a separate branch off `main` for each new feature/fix to be worked on
- Use clear and consistent branch names, e.g., _{#issue}-{feature-description}_
- Feature branches are merged into `main` using pull/merge requests
- Whoever merges into `main` is responsible for adding appropriate notes to the [CHANGELOG][changelog]
- We never directly push changes to `main`
- Avoid branching off from another feature branch

```mermaid
%%{init: { 'theme': 'base', 'gitGraph': {'showCommitLabel': false}} }%%
gitGraph
  commit
  commit
  branch 42-some-feature
  checkout 42-some-feature
  commit
  commit
  checkout main
  merge 42-some-feature
  commit
  branch 43-another-feature
  checkout 43-another-feature
  commit
  commit
  commit
  checkout main
  merge 43-another-feature
  commit
```

## Local Development Environment

You can (and should) run our test suite using [tox][tox].
For a more traditional environment we recommend to develop using a Python 3.10 release.

Create a new virtual environment using your favorite environment manager.
Then get an up to date checkout of the `eurocropsml` repository:

```console
$ git clone https://github.com/dida-do/eurocropsml.git
```

Change into the newly created directory and **activate your virtual environment** if you have not done that already.


### Makefile Installation and Testing (Recommended)

For convenience we have configured several commands that are frequently used during development into a `Makefile`.

You can install and editable version of the `eurocropsml` package with all its development dependencies:

```console
$ make install
```

You can run our test suite against multiple Python versions:
```console
$ make test
```
*(Under the hood we use `tox` for this, see below.)*

To avoid committing code not following our style guide, we advise you to always run our automated code formatting and linting after you have made changes:
```console
$ make format lint
```

You can use
```console
$ make help
```
to see a list of all available `Makefile` commands.


### Manual Installation and Testing (Not Recommended)

Of course you can also run the installation, testing, formatting, and linting commands manually, e.g., in case you want control over extra command line options. 

The alternative way to install an editable version of the `eurocropsml` package along with all its development dependencies is:

```console
$ pip install -e '.[dev]'
```

Now you should be able to run tests against a single (currently active) Python version manually

```console
$ python -m pytest
```

or use our configured [tox][tox] environments, e.g.,

```console
$ tox run -e py310,py311
```

to run tests against different Python versions 3.10 and 3.11.

*(Similarly corresponding `tox` commands are also used by our `Makefile` under the hood.)*

## Code Style and Formatting
We generally follow [PEP 8][PEP8], [PEP 257][PEP257], and [PEP 484][PEP484] to the extent that it is checked/enforced by our configured [black][black] code formatter, [flake8][flake8] linter, and [mypy][mypy] static type checker. 
We use [isort][isort] to sort all imports.

The formatting can be automated.
If you run our `make` or [tox][tox] commands before committing (see above), you do not have to spend any thoughts or time on formatting your code at all.

### Docstrings

- We use [google style][googledocs] docstrings and [PEP 484][PEP484] type hints for type annotations:
    ```python
    def function_with_type_annotations(param1: int, param2: str) -> bool:
        """Example function with type annotations.

        An optional more detailed description of the function behavior.

        Args:
            param1: The first parameter.
            param2: The second parameter.
        Returns:
            The return value. True for success, False otherwise.
        """
        ...
    ```
    ```python
    class ExampleClass(object):
        """The summary line for a class docstring should fit on one line.

        If the class has public attributes, they may be documented here in an ``Attributes`` section and follow the same formatting as a function's ``Args`` section. 

        Attributes:
            attr1: Description of `attr1`.
            attr2: Description of `attr2`.

        Args:
            param1: Description of `param1`.
            param2: Description of `param2`.
                Multiple lines are supported as well.
            param3: Description of `param3`.

        """

        def __init__(self, param1: str, param2: int, param3: float):
            ...
    ```
- If you make additions or changes to the public interface part of the package, tag the respective docstrings with  `..  versionadded:: XX.YY.ZZ <NOTE>` or `.. versionchanged:: XX.YY.ZZ <NOTE>` directives.

### Naming Conventions

We generally adhere to using

- snake case `lower_case_with_underscores` for variables and functions
- camel case `CapitalizedWords` for classes
- capitalization `UPPER_CASE_WITH_UNDERSCORES` for constants and (global) configurations

Functions and methods should typically have names describing the effect/action that they perform, i.e. verbs as names, e.g., `def parse_some_thing(...)`.

Classes and instances should typically have names describing the type of object, i.e. nouns as names, e.g., `class SomeThingParser(...)`.


## Tests

- Write assertions as `actual == expected` for consistency:
  ```python
  x = f(...)

  assert x.important_attribute == 42
  assert x.another_attribute == "foobar"
  ```
- Use our `make` or [tox][tox] commands (see above) to run our tests.
  It will ensure the test suite runs with all the correct dependencies against all supported Python versions just as it will in our CI.
  If you lack Python versions, you can can limit the environments like `tox -e py310,py311`.
- Write docstrings for your tests. Here are tips for writing [good test docstrings][gooddocstrings].


## Documentation

- We use [Sphinx][sphinx] to build our documentation.
- Documentation files can be written in the [reStructuredText][rst] or [Markdown][md] format.
- Use [semantic newlines][semanticnewlines] in both cases:
  ```md
  This is a sentence.
  This is another sentence.
  ```

## Governance

`eurocropsml` was created as part of a research project and is
maintained by volunteers.
We are always open to new members that want to help.
Just let us know if you want to join the team.

**Everyone is welcome to help review pull/merge requests of others but nobody should review and merge their own code.**

---

Please note that this project is released with a Contributor [Code of Conduct][code-of-conduct].
By participating in this project you agree to abide by its terms.
Please report any harm to the project team in any way you find appropriate.

Thank you again for considering contributing to `eurocropsml`!


[PEP8]: https://www.python.org/dev/peps/pep-0008/
[PEP257]: https://www.python.org/dev/peps/pep-0257/
[PEP484]: https://www.python.org/dev/peps/pep-0484/
[gooddocstrings]: https://jml.io/pages/test-docstrings.html
[code-of-conduct]: https://github.com/dida-do/eurocropsml/blob/main/CODE_OF_CONDUCT.md
[changelog]: https://github.com/dida-do/eurocropsml/blob/main/CHANGELOG.md
[tox]: https://tox.readthedocs.io/
[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[rst]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[md]: https://www.sphinx-doc.org/en/master/usage/markdown.html
[semanticnewlines]: https://rhodesmill.org/brandon/2012/one-sentence-per-line/
[black]: https://github.com/psf/black
[isort]: https://github.com/timothycrosley/isort
[googledocs]: https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings
[flake8]: https://flake8.pycqa.org/en/latest/
[mypy]: https://mypy.readthedocs.io/en/stable/