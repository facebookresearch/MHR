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

"""Download the MHR release assets.

Without --lod, the command downloads the release's assets.zip (FBX rigs,
pose correctives and the TorchScript model) into --dest, which then holds an
assets/ folder. With --lod, it downloads the converted assets for the
selected LODs, checks them against the release manifest, and extracts them
into <dest>/assets/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

from pathlib import Path, PureWindowsPath

from ._rig_schema import LODS, RIG_SCHEMA_VERSION

DEFAULT_REPO = "facebookresearch/MHR"
DEFAULT_RELEASE = "latest"
DEFAULT_ARCHIVE = "assets.zip"
DEFAULT_MANIFEST = f"mhr-assets-v{RIG_SCHEMA_VERSION}.json"
ASSET_FOLDER = "assets"
CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 60
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _asset_url(repo: str, release: str, archive: str) -> str:
    if release == "latest":
        return f"https://github.com/{repo}/releases/latest/download/{archive}"
    return f"https://github.com/{repo}/releases/download/{release}/{archive}"


def _download(
    url: str, output: Path, retries: int, timeout: float = TIMEOUT_SECONDS
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mhr-assets"})
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                output.open("wb") as f,
            ):
                while chunk := response.read(CHUNK_SIZE):
                    f.write(chunk)

            if output.stat().st_size == 0:
                raise RuntimeError("downloaded archive is empty")
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            output.unlink(missing_ok=True)
            # A client error other than a timeout or a rate limit does not go
            # away on a retry.
            if (
                isinstance(exc, urllib.error.HTTPError)
                and 400 <= exc.code < 500
                and exc.code not in (408, 429)
            ):
                break
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def _validate_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            first_bad_file = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"{path} is not a valid zip archive") from exc

    if first_bad_file is not None:
        raise RuntimeError(f"{path} is corrupt at {first_bad_file}")


def _member_target(dest: Path, name: str) -> Path:
    """Return where an archive member goes, or raise if its name is unsafe."""

    parts = name.rstrip("/").split("/")
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or PureWindowsPath(name).drive
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise RuntimeError(f"unsafe archive member name: {name!r}")
    root = dest.resolve()
    target = (dest / name).resolve()
    if root not in target.parents:
        raise RuntimeError(f"archive member escapes destination: {name}")
    return target


def _safe_members(zf: zipfile.ZipFile, dest: Path) -> list[zipfile.ZipInfo]:
    """Check every member name before anything is extracted."""

    members = zf.infolist()
    for member in members:
        _member_target(dest, member.filename)
    return members


def _extract(path: Path, dest: Path, member: str | None, output: Path | None) -> None:
    with zipfile.ZipFile(path) as zf:
        if member is not None:
            try:
                source = zf.open(member)
            except KeyError as exc:
                raise RuntimeError(f"archive does not contain {member}") from exc

            target = output if output is not None else dest / Path(member).name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as f:
                shutil.copyfileobj(source, f)
            return

        for archive_member in _safe_members(zf, dest):
            zf.extract(archive_member, dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _lod(value: str) -> int:
    parsed = int(value)
    if parsed not in LODS:
        raise argparse.ArgumentTypeError(f"must be between {LODS[0]} and {LODS[-1]}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MHR release assets.")
    parser.add_argument(
        "--repo", default=os.environ.get("MHR_ASSETS_REPO", DEFAULT_REPO)
    )
    parser.add_argument(
        "--release", default=os.environ.get("MHR_ASSETS_RELEASE", DEFAULT_RELEASE)
    )
    parser.add_argument(
        "--archive",
        help=f"Archive to download without --lod. Defaults to {DEFAULT_ARCHIVE}.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("MHR_ASSETS_DEST", ".")),
        help="Directory for the downloaded archives; it receives the assets/ folder.",
    )
    converted = parser.add_mutually_exclusive_group()
    converted.add_argument(
        "--lod",
        action="append",
        type=_lod,
        help="Download the converted assets for this LOD (repeatable).",
    )
    converted.add_argument(
        "--all-lods",
        action="store_true",
        help="Download the converted assets for every LOD.",
    )
    parser.add_argument(
        "--member",
        help="Extract only one archive member, for example assets/mhr_model.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path when --member is used. Defaults to --dest / basename.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download and validate the archives without extracting them.",
    )
    parser.add_argument(
        "--retries",
        type=_positive_int,
        default=_positive_int(os.environ.get("MHR_ASSETS_RETRIES", "3")),
    )
    args = parser.parse_args(argv)
    if (args.lod or args.all_lods) and (args.member or args.output or args.archive):
        parser.error("--archive, --member and --output apply only without --lod")
    if args.archive is None:
        args.archive = os.environ.get("MHR_ASSETS_ARCHIVE", DEFAULT_ARCHIVE)
    return args


def _manifest_entry(key: str, entry: object) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise RuntimeError(f"invalid {key} entry in {DEFAULT_MANIFEST}")
    filename, checksum = entry.get("filename"), entry.get("sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".zip")
        or filename == ".zip"
        or not isinstance(checksum, str)
        or not _SHA256.fullmatch(checksum)
    ):
        raise RuntimeError(f"invalid {key} entry in {DEFAULT_MANIFEST}")
    return filename, checksum


def _install(bundles: list[Path], assets: Path) -> None:
    """Extract the bundles into a staging folder next to assets, then move
    the files into assets.

    The staging folder is on the same file system as assets, so each move is
    a rename. Nothing in assets changes unless every bundle extracted.
    """

    assets.mkdir(parents=True, exist_ok=True)
    real_assets = assets.resolve()
    with tempfile.TemporaryDirectory(
        dir=real_assets.parent, prefix=f".{real_assets.name}.staging."
    ) as tmp:
        staging = Path(tmp)
        files: dict[Path, Path] = {}
        for index, bundle in enumerate(bundles):
            folder = staging / str(index)
            _extract(bundle, folder, None, None)
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(folder)
                if relative in files:
                    # Every bundle holds the same LICENSE.txt.
                    if files[relative].read_bytes() != path.read_bytes():
                        raise RuntimeError(
                            f"bundles disagree on the contents of {relative}"
                        )
                    continue
                files[relative] = path
        for relative in files:
            if (real_assets / relative).is_dir():
                raise RuntimeError(
                    f"{real_assets / relative} is a directory; remove it and retry"
                )
        for relative, source in files.items():
            target = real_assets / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)


def _download_converted(args: argparse.Namespace) -> None:
    lods = list(LODS) if args.all_lods else sorted(set(args.lod))
    with tempfile.TemporaryDirectory(dir=args.dest, prefix=".mhr-assets.") as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / DEFAULT_MANIFEST
        url = _asset_url(args.repo, args.release, DEFAULT_MANIFEST)
        print(f"Downloading {url}")
        try:
            _download(url, manifest_path, args.retries)
        except RuntimeError as exc:
            cause = exc.__cause__
            if isinstance(cause, urllib.error.HTTPError) and cause.code == 404:
                raise RuntimeError(
                    f"release {args.release} of {args.repo} has no converted MHR "
                    f"assets ({DEFAULT_MANIFEST} not found); omit --lod and "
                    f"--all-lods to download {DEFAULT_ARCHIVE}"
                ) from exc
            raise
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError as exc:
            raise RuntimeError(f"{DEFAULT_MANIFEST} is not valid JSON") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != RIG_SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"unsupported MHR asset manifest schema in {DEFAULT_MANIFEST}"
            )
        available = manifest.get("lods")
        if not isinstance(available, dict) or not all(
            isinstance(key, str)
            and key.isascii()
            and key.isdecimal()
            and str(int(key)) == key
            for key in available
        ):
            raise RuntimeError(f"invalid lods entry in {DEFAULT_MANIFEST}")
        missing = [lod for lod in lods if str(lod) not in available]
        if missing:
            raise RuntimeError(
                f"release {args.release} has no converted assets for LOD "
                f"{', '.join(map(str, missing))}; it has LOD "
                f"{', '.join(sorted(available, key=int)) or 'none'}"
            )
        entries = [_manifest_entry("common", manifest.get("common"))]
        entries += [_manifest_entry(f"LOD {lod}", available[str(lod)]) for lod in lods]
        filenames = [filename for filename, _ in entries]
        if len(set(filenames)) != len(filenames):
            raise RuntimeError(f"{DEFAULT_MANIFEST} lists a bundle twice")

        # Download and check every bundle before anything is extracted.
        bundles = []
        for filename, checksum in entries:
            bundle = tmp_dir / filename
            url = _asset_url(args.repo, args.release, filename)
            print(f"Downloading {url}")
            _download(url, bundle, args.retries)
            if _sha256(bundle) != checksum:
                raise RuntimeError(f"checksum mismatch for {filename}")
            _validate_zip(bundle)
            with zipfile.ZipFile(bundle) as zf:
                _safe_members(zf, args.dest / ASSET_FOLDER)
            bundles.append(bundle)

        if not args.no_extract:
            _install(bundles, args.dest / ASSET_FOLDER)
        for bundle in bundles:
            bundle.replace(args.dest / bundle.name)
            print(f"Saved {args.dest / bundle.name}")
        if not args.no_extract:
            print(f"Extracted the converted assets into {args.dest / ASSET_FOLDER}")


def _download_legacy(args: argparse.Namespace) -> None:
    archive_path = args.dest / args.archive
    url = _asset_url(args.repo, args.release, args.archive)

    with tempfile.NamedTemporaryFile(
        prefix=f"{args.archive}.", suffix=".tmp", dir=args.dest, delete=False
    ) as f:
        tmp_path = Path(f.name)

    try:
        print(f"Downloading {url}")
        _download(url, tmp_path, args.retries)
        _validate_zip(tmp_path)
        tmp_path.replace(archive_path)
        print(f"Saved {archive_path}")

        if not args.no_extract:
            _extract(archive_path, args.dest, args.member, args.output)
            print(f"Extracted {archive_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    try:
        if args.lod or args.all_lods:
            _download_converted(args)
        else:
            _download_legacy(args)
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
