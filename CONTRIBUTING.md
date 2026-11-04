# Contributing to Kriterion Lab

Kriterion Lab is an academic reference implementation. Contributions should preserve its central property: every result must remain reproducible and easy to explain in a classroom or project defense.

## Development setup

1. Fork and clone the repository.
2. Create a Python 3.11 or newer virtual environment.
3. Install the package with `pip install -e ".[dev]"`.
4. Run `pytest` and `ruff check .` before opening a pull request.
5. Start the application with `streamlit run app.py`.

## Contribution standards

- Keep random seeds explicit and datasets publicly reproducible.
- Add tests for mathematical changes and metric calculations.
- Preserve scratch implementations alongside library comparisons.
- Avoid dependencies that are not needed at runtime.
- Keep visual encodings accessible and consistent with the Zinc theme.
- Do not present the application as a medical diagnostic device.
- Update the report, presentation source, and exported experiment data when results change.

## Pull requests

Describe the motivation, the affected experiment, and the commands used for verification. Include before-and-after metric tables or visual evidence when behavior changes. Small, focused pull requests are preferred.

## Data and research ethics

Do not add private, identifiable, or restricted clinical data. New datasets must include provenance, license information, target definition, and known limitations.

