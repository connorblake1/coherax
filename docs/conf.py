"""Sphinx configuration for coherax documentation."""

project = "coherax"
copyright = "2026, Connor Blake and Liang Jiang"
author = "Connor Blake, Liang Jiang"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Avoid importing heavy C extensions during doc build
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "dynamiqs",
    "equinox",
    "optax",
    "jaxtyping",
    "strawberryfields",
    "cma",
]
