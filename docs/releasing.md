# Releasing

Vui follows [Semantic Versioning](https://semver.org/). Released versions get
a git tag (`vX.Y.Z`) and a corresponding GitHub release.

## Versioning policy

- **Major** (`X.0.0`) — breaking changes: model format, codec, public Python
  API (`vui.engine`, `vui.model`), realtime API event shape, CLI entry points,
  on-disk paths (`~/.vui/`).
- **Minor** (`x.Y.0`) — new features, new endpoints, additive config or
  conditioning fields, new ASR/LLM backends.
- **Patch** (`x.y.Z`) — bug fixes, dependency bumps without API impact, docs.

A new model checkpoint with a different architecture or codec is a major bump
even if the Python API is unchanged — users have to re-download weights.

## Source of truth

The version is declared in two places and must be kept in sync:

- `pyproject.toml` → `[project] version`
- `src/vui/__init__.py` → `__version__`

Both should read the same string. A CI check or single-source refactor would
be nice — see open work below.

## Cutting a release

1. **Pick the version.** Decide major/minor/patch per the policy above.
2. **Bump the version** in both files.
3. **Update `CHANGELOG.md`.** Move items out of `Unreleased` (if any) into a
   new `[X.Y.Z] - YYYY-MM-DD` section. Add the comparison link at the bottom.
   Use the headings `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` /
   `Security`.
4. **Sanity-check the build.**
   ```sh
   uv sync
   uv run python -c "import vui; print(vui.__version__)"
   uv run python -m vui.serving.stream --help
   ```
5. **Commit.**
   ```sh
   git add pyproject.toml src/vui/__init__.py CHANGELOG.md
   git commit -m "Release vX.Y.Z"
   ```
6. **Tag and push.**
   ```sh
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main --follow-tags
   ```
7. **Create the GitHub release.** Notes come from the CHANGELOG entry:
   ```sh
   gh release create vX.Y.Z \
       --title "vX.Y.Z" \
       --notes-file <(awk '/^## \[X.Y.Z\]/,/^## \[/' CHANGELOG.md | sed '$d')
   ```
   Or paste the CHANGELOG section into the release body via the web UI.
8. **Verify.**
   - `gh release list` shows the new release.
   - `git describe --tags` on `main` reports `vX.Y.Z`.

## Model weights

Code releases and weight releases are decoupled. The streaming server
auto-downloads weights from <https://huggingface.co/fluxions/vui> on first run.
If a release requires new weights, push the weights to the HF repo *before*
tagging, and call out the dependency in the CHANGELOG `Changed` section.

## Hotfixes

For a patch release off an older line (e.g. `1.0.x` while `main` is on `1.1`):

1. Branch from the tag: `git switch -c release/1.0.x v1.0.0`.
2. Cherry-pick the fix.
3. Bump to `1.0.1`, update CHANGELOG, tag `v1.0.1`, release.
4. Forward-port the fix to `main`.

## Open work

- Single-source the version (read `pyproject.toml` via `importlib.metadata`
  from `__init__.py`) so step 2 only touches one file.
- Optional `.github/workflows/release.yml` to auto-create the GitHub release
  from CHANGELOG on tag push, and (if we ever publish to PyPI) build + upload
  the wheel.
