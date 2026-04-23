# Sphinx configuration — minimal baseline.
# Extensions, theme, autodoc, etc. are layered in later tasks.

project = "Weight, what?"
author = "TAP team"
copyright = "2026, TAP team"
release = "0.1.0"

extensions: list[str] = []
exclude_patterns = ["_build", ".DS_Store"]
html_theme = "alabaster"  # Sphinx default; replaced with Furo in Task 3.
