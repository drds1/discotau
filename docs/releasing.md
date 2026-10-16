# Releasing discotau to PyPI

How to publish a new version of `discotau` to
[PyPI](https://pypi.org/project/discotau/). Every release is a GitHub
release: publishing one triggers
[`.github/workflows/publish.yml`](https://github.com/drds1/discotau/blob/main/.github/workflows/publish.yml), which
builds the package and uploads it to PyPI. Nobody uploads by hand, and no
API token is stored anywhere.

## Contents

- [One-off setup](#one-off-setup)
- [Choosing the version number](#choosing-the-version-number)
- [Where the version number lives](#where-the-version-number-lives)
- [Release checklist](#release-checklist)
- [Checking the release worked](#checking-the-release-worked)
- [When something goes wrong](#when-something-goes-wrong)

## One-off setup

This has to happen once, before the first release. Skip it if
https://pypi.org/project/discotau/ already exists and lists `drds1/discotau`
under "Trusted publishers" in the project's settings.

PyPI needs to know it can trust uploads coming from this repository's
workflow ("trusted publishing"):

1. Log in at https://pypi.org (two-factor authentication is mandatory).
2. Go to https://pypi.org/manage/account/publishing/ and, under
   **Add a new pending publisher**, pick the **GitHub** tab.
3. Enter these values exactly, then click **Add**:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `discotau` |
   | Owner | `drds1` |
   | Repository name | `discotau` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

A pending publisher does not reserve the name: the project only exists, and
belongs to you, once the first upload succeeds. Cut the first release soon
after adding it.

GitHub creates the `pypi` environment the first time the workflow runs. If
you want releases to need a manual approval before uploading, add a
required reviewer to it afterwards (repository **Settings → Environments →
pypi**).

## Choosing the version number

Versions follow [semantic versioning](https://semver.org/),
`MAJOR.MINOR.PATCH`:

- **PATCH** (`0.1.0` → `0.1.1`): bug fixes only; existing code and results
  keep working.
- **MINOR** (`0.1.1` → `0.2.0`): new features, or changes to defaults. While
  the major version is `0`, a minor bump may also break the API; say so in
  the release notes.
- **MAJOR** (`0.x` → `1.0.0`): the API is declared stable; after that, only
  a major bump may break it.

PyPI never accepts the same version twice, even if the earlier upload was
deleted, so every release needs a new number.

## Where the version number lives

Two files, and both must be bumped together:

| File | Line | Why it is there |
|---|---|---|
| `pyproject.toml` | `version = "0.1.0"` | The package version PyPI and `pip` see. The single source of truth. |
| `CITATION.cff` | `version: 0.1.0` | What GitHub's "Cite this repository" button shows. |

`tests/test_version_consistency.py` fails if the two disagree.
`discotau.__version__` is not a third copy: it reads the installed
package's metadata, which comes from `pyproject.toml`. In a local
development install it only updates after you reinstall (`poetry install`),
so a freshly bumped checkout can still report the old number until then.

## Release checklist

Run from the repository root, on an up-to-date `main`:

```bash
git switch main && git pull
git status            # must be clean
```

1. **Run the full test suite.** CI doesn't run automatically on push (see
   the note at the top of `.github/workflows/tests.yml`), so trigger it and
   wait for it to pass:

   ```bash
   gh workflow run tests.yml --ref main
   gh run watch
   ```

   Or run it locally (about 15 minutes):
   `MPLBACKEND=Agg poetry run pytest`.

2. **Bump the version** in `pyproject.toml` and `CITATION.cff` (see above),
   then check the two agree:

   ```bash
   poetry run pytest tests/test_version_consistency.py
   ```

3. **Check the package builds** and that PyPI will accept its metadata and
   README:

   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check --strict dist/*
   rm -rf dist build
   ```

4. **Commit and push the bump:**

   ```bash
   git commit -am "Release v0.2.0"
   git push
   ```

5. **Publish the GitHub release.** The tag is the version with a `v` in
   front, and must match `pyproject.toml`:

   ```bash
   gh release create v0.2.0 --target main --title "v0.2.0" --generate-notes
   gh run watch
   ```

   From the website instead: **Releases → Draft a new release**, type the
   tag under "Choose a tag" and choose "Create new tag on publish", target
   `main`, click **Generate release notes**, edit them if needed, then
   **Publish release**.

   `--generate-notes` lists the merged pull requests since the previous
   release. Add a short summary at the top of anything a user needs to
   act on: changed defaults, renamed arguments, results that will differ.

## Checking the release worked

- In the **Actions** tab, "Publish to PyPI" shows both jobs (build,
  publish) green.
- https://pypi.org/project/discotau/ shows the new version, with the README
  (including the animation) rendered.
- A clean install picks it up:

  ```bash
  python -m venv /tmp/discotau-check
  /tmp/discotau-check/bin/pip install discotau==0.2.0
  /tmp/discotau-check/bin/python -c "import discotau; print(discotau.__version__)"
  rm -rf /tmp/discotau-check
  ```

  PyPI's index can take a minute or two to show a new version.

## When something goes wrong

- **Publish job fails with "invalid-publisher" or a trusted-publishing
  error**: the pending/trusted publisher on PyPI doesn't exactly match the
  workflow. Check the owner, repository name (`discotau`, not the old
  `echofit`), workflow file name (`publish.yml`) and environment (`pypi`).
  Fix it on PyPI, then re-run the failed job from the Actions tab; there's
  no need to make a new release.
- **"File already exists"**: that version is already on PyPI. Bump the
  version (a new PATCH number is fine), commit, and cut a new release. Delete
  the failed GitHub release and its tag first if you don't want it left
  behind: `gh release delete v0.2.0 --cleanup-tag`.
- **Released from the wrong commit, or forgot to bump the version before
  tagging**: if the publish job failed, delete the release and tag (as
  above), fix `main`, and release again. If it succeeded, the upload can't
  be replaced; release a new PATCH version instead.
- **A broken version made it to PyPI**: *yank* it (on PyPI, the project's
  **Manage → Releases → Options → Yank**). `pip install discotau` then skips
  it, while anyone who pinned exactly that version can still install it.
  Then release a fixed PATCH version. Don't delete it: the number can never
  be reused anyway.
