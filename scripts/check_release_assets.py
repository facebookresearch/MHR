#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that a release's converted assets come from its assets.zip.

The check downloads the release's assets.zip and its converted bundles
(checked against their manifest by mhr-download-assets), then:

1. converts assets.zip again with scripts/convert_assets.py;
2. checks that the release manifest lists the same source files (with the
   same SHA-256) and the same LODs as the new conversion;
3. checks that every bundle holds exactly its NPZ file and the LICENSE.txt of
   assets.zip, and compares every array of the NPZ file with the new
   conversion.

Integer, boolean and string arrays must be equal. Float arrays may differ by
float32 rounding (FLOAT_ATOL, FLOAT_RTOL), because the PyMomentum version that
converts them can differ from the one that made the release. The check needs
about 9 GB of disk. Exits with status 1 if a check fails. Run it from the
repository root in the Pixi environment, which includes PyMomentum:

    pixi run python -m scripts.check_release_assets --release v1.2.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile

from pathlib import Path

import numpy as np

from mhr import download_assets
from mhr._rig_schema import LODS
from scripts.convert_assets import convert_assets

# Conversions with PyMomentum 0.1.95 and 0.1.114 differ by at most 6e-8 (one
# float32 step near 1); allow a few steps relative to the value as well.
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-6
# Recorded by the converter; it may name another PyMomentum version.
IGNORED_ARRAYS = {"source_momentum_version"}
LICENSE = "LICENSE.txt"


def compare_npz(expected: Path, actual: Path, label: str) -> list[str]:
    """Return the differences between two NPZ files, or an empty list."""

    problems = []
    with (
        np.load(expected, allow_pickle=False) as want,
        np.load(actual, allow_pickle=False) as got,
    ):
        want_names = set(want.files) - IGNORED_ARRAYS
        got_names = set(got.files) - IGNORED_ARRAYS
        if want_names != got_names:
            return [f"{label}: arrays differ: {sorted(want_names ^ got_names)}"]
        for name in sorted(want_names):
            a, b = want[name], got[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                problems.append(
                    f"{label}: {name} is {b.dtype}{list(b.shape)}, "
                    f"expected {a.dtype}{list(a.shape)}"
                )
            elif a.dtype.kind == "f":
                close = np.isclose(
                    a, b, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True
                )
                if not close.all():
                    difference = float(np.max(np.abs(a - b)[~close]))
                    problems.append(
                        f"{label}: {name} differs by up to {difference:.3g}"
                    )
            elif not np.array_equal(a, b):
                problems.append(f"{label}: {name} differs")
    return problems


def compare_manifests(expected: dict, actual: dict) -> list[str]:
    """Compare the release manifest with the one of the new conversion."""

    problems = []
    if actual.get("source_sha256") != expected["source_sha256"]:
        problems.append("the manifest's source files do not match assets.zip")
    if set(actual.get("lods", {})) != set(expected["lods"]):
        problems.append(
            f"the manifest lists LODs {sorted(actual.get('lods', {}))}, "
            f"expected {sorted(expected['lods'])}"
        )
    return problems


def check_bundle(
    bundle: Path, npz_name: str, expected_npz: Path, license: bytes, work: Path
) -> list[str]:
    """Check that a release bundle holds exactly npz_name and LICENSE.txt, with
    the expected contents."""

    with zipfile.ZipFile(bundle) as archive:
        names = sorted(archive.namelist())
        if names != sorted([npz_name, LICENSE]):
            return [f"{bundle.name} holds {names}, expected {npz_name} and {LICENSE}"]
        problems = []
        if archive.read(LICENSE) != license:
            problems.append(f"{bundle.name}: {LICENSE} differs from assets.zip")
        extracted = Path(archive.extract(npz_name, work))
    try:
        problems += compare_npz(expected_npz, extracted, f"{bundle.name}: {npz_name}")
    finally:
        extracted.unlink()
    return problems


def check_release(repo: str, release: str, work: Path) -> list[str]:
    legacy = work / "legacy"
    converted = work / "converted"
    published = work / "published"

    download_assets.main(["--repo", repo, "--release", release, "--dest", str(legacy)])
    print("Converting assets.zip again")
    expected_manifest = json.loads(
        convert_assets(legacy / "assets", converted, LODS, "check").read_text()
    )
    license = (legacy / "assets" / LICENSE).read_bytes()
    shutil.rmtree(legacy)  # free the disk before the next download

    # Checks every bundle against the release manifest, without extracting.
    download_assets.main(
        [
            "--repo",
            repo,
            "--release",
            release,
            "--all-lods",
            "--no-extract",
            "--dest",
            str(published),
        ]
    )
    manifest_path = work / download_assets.DEFAULT_MANIFEST
    download_assets._download(
        download_assets._asset_url(repo, release, download_assets.DEFAULT_MANIFEST),
        manifest_path,
        retries=3,
    )
    manifest = json.loads(manifest_path.read_text())

    problems = compare_manifests(expected_manifest, manifest)
    bundles = [(manifest["common"]["filename"], "rig.npz")]
    bundles += [
        (manifest["lods"][str(lod)]["filename"], f"lod{lod}.npz")
        for lod in LODS
        if str(lod) in manifest.get("lods", {})
    ]
    for filename, npz_name in bundles:
        problems += check_bundle(
            published / filename, npz_name, converted / npz_name, license, work
        )
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check that a release's converted assets come from its assets.zip."
    )
    parser.add_argument("--repo", default=download_assets.DEFAULT_REPO)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--work",
        type=Path,
        help="Folder for the downloads; a temporary one by default.",
    )
    args = parser.parse_args(argv)

    if args.work is not None:
        args.work.mkdir(parents=True, exist_ok=True)
        if any(args.work.iterdir()):
            parser.error(f"--work folder {args.work} is not empty")
        problems = check_release(args.repo, args.release, args.work)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            problems = check_release(args.repo, args.release, Path(tmp))
    if problems:
        print(f"The converted assets of {args.release} do not match its assets.zip:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print(f"The converted assets of {args.release} match its assets.zip.")


if __name__ == "__main__":
    main()
