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

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

from pathlib import Path
from typing import Iterable

DEFAULT_REPO = "facebookresearch/MHR"
DEFAULT_RELEASE = "latest"
DEFAULT_ARCHIVE = "assets.zip"
CHUNK_SIZE = 1024 * 1024


def _asset_url(repo: str, release: str, archive: str) -> str:
    if release == "latest":
        return f"https://github.com/{repo}/releases/latest/download/{archive}"
    return f"https://github.com/{repo}/releases/download/{release}/{archive}"


def _download(url: str, output: Path, retries: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mhr-assets"})
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request) as response, output.open("wb") as f:
                while chunk := response.read(CHUNK_SIZE):
                    f.write(chunk)

            if output.stat().st_size == 0:
                raise RuntimeError("downloaded archive is empty")
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            output.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"failed to download {url}: {last_error}")


def _validate_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            first_bad_file = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"{path} is not a valid zip archive") from exc

    if first_bad_file is not None:
        raise RuntimeError(f"{path} is corrupt at {first_bad_file}")


def _safe_members(zf: zipfile.ZipFile, dest: Path) -> Iterable[zipfile.ZipInfo]:
    root = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"archive member escapes destination: {member.filename}")
        yield member


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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MHR release assets.")
    parser.add_argument("--repo", default=os.environ.get("MHR_ASSETS_REPO", DEFAULT_REPO))
    parser.add_argument(
        "--release", default=os.environ.get("MHR_ASSETS_RELEASE", DEFAULT_RELEASE)
    )
    parser.add_argument(
        "--archive", default=os.environ.get("MHR_ASSETS_ARCHIVE", DEFAULT_ARCHIVE)
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("MHR_ASSETS_DEST", ".")),
        help="Directory for the downloaded archive and extracted files.",
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
        help="Download and validate the archive without extracting it.",
    )
    parser.add_argument(
        "--retries",
        type=_positive_int,
        default=_positive_int(os.environ.get("MHR_ASSETS_RETRIES", "3")),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)

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
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
