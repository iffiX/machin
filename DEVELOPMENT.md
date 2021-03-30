# Code style
The code follows code styling by [black](https://github.com/psf/black).

To automate code formatting, [pre-commit](https://github.com/pre-commit/pre-commit) is used, to run code checks before commiting changes.
If you have pre-commit installed from the requirements-dev.txt simple run ``pre-commit install`` to install the hooks for this repo.

# Setup

### git setup for ignoring refactoring commits
Run `git config --local include.path ../.gitconfig` inside your clone of the repo. 
Update git to `2.23` or higher to be able to use this feature.
