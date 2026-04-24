"""Sphinx configuration for the TAP docs site.

The docs env composes the `cuda` Pixi feature so autodoc imports the
real torch/transformers/ai2-olmo-core at build time — no mock imports
and no sys.path gymnastics required.
"""

from __future__ import annotations

project = "Weight, what?"
author = "TAP team"
copyright = "2026, TAP team"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

# MyST extras.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", ".DS_Store"]

# Autodoc / autosummary.
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Napoleon — project uses Google-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = True

# Intersphinx — cross-link to external projects.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "hf": ("https://huggingface.co/docs/transformers/main/en", None),
}

html_theme = "furo"
html_title = "Weight, what?"
templates_path = ["_templates"]
