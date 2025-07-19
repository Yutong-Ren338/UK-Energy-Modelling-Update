This project uses `uv` to manage dependencies and run tasks.

## Setup

To get started, install `uv`:

```bash
pip install uv
```

## Running Tests

To run the test suite, use the following command:

```bash
uv run pytest
```

## Linting

This project uses `ruff` for linting and formatting.

To check for linting errors, run:

```bash
uv run ruff check .
```

To format the code, run:

```bash
uv run ruff format .
```

You can apply safe autofixes from `ruff` to automatically correct some linting issues.

## Code Style

### Docstrings

- Docstrings should follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Trivial functions and methods do not require docstrings.
- Type hints should be included in the function signature, not in the docstring.
