# Sphinx configuration for the TAP docs site.

project = "Weight, what?"
author = "TAP team"
copyright = "2026, TAP team"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

# MyST extras.
myst_enable_extensions = [
    "colon_fence",   # ::: fences for directives
    "deflist",       # definition lists
    "linkify",       # bare URLs become links
    "substitution",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", ".DS_Store"]

html_theme = "furo"
html_title = "Weight, what?"
templates_path = ["_templates"]
