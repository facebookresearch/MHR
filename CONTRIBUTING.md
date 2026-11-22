# Contributing to MHR
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Converting the release assets

`scripts/convert_assets.py` converts the FBX and NPZ release assets into the
portable tensor format described in `mhr/_rig_schema.py`, and writes the
per-LOD release bundles and their manifest. Run it in the Pixi environment,
which includes PyMomentum:

```bash
pixi run download-assets
pixi run python scripts/convert_assets.py --assets assets --output dist/assets
```

Running it twice in the same environment gives byte-identical output. The
bytes can change with the PyMomentum version, which the script records in
`rig.npz`, and with the NumPy, Python and zlib versions that write the files.

## Releasing

Every release must also work for the MHR versions that are already
installed. Before publishing a release:

1. Convert the assets of the release as above, twice, and check that the
   SHA-256 values of the two outputs match.
2. Attach all of these to the GitHub release:
   - `assets.zip`, with its current layout. Older wheels, `mhr-download-assets`
     without `--lod`, and `pixi run download-assets` download it.
   - `mhr-assets-v1.json`, `mhr-assets-v1-common.zip`, and every
     `mhr-assets-v1-lod<N>.zip`. `mhr-download-assets --lod` downloads them.
3. Create the release as a draft, attach the files, then publish it. Its tag
   starts the publish workflow. Before it uploads the wheel, the workflow
   downloads the release's `assets.zip` and converted assets, checks them
   against the manifest, converts `assets.zip` again and compares the arrays
   (`scripts/check_release_assets.py`). A release whose converted assets are
   missing or do not come from its `assets.zip` is not published. To check a
   release by hand, run the "Check release assets" workflow or
   `pixi run python -m scripts.check_release_assets --release <tag>`.
4. Check the published release:
   `mhr-download-assets --release <tag> --all-lods`, then
   `MHR_TEST_ASSETS=assets pixi run pytest scripts/tests/test_pure_torch_runtime.py`.

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to MHR, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
