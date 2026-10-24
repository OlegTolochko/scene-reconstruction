"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
import time
from pathlib import Path
import urllib.request
from urllib.request import urlretrieve
import urllib.error
import yaml
from munch import Munch
from tqdm import tqdm
import json
import os
import re
import sys
import zipfile

from scene_release import ScannetppScene_Release


# --- pretty, colorized terminal error messages ---------------------------
class _Ansi:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def _use_color():
    # Only emit ANSI when writing to a real terminal that wants it.
    return (
        sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM", "") != "dumb"
    )


def _c(code, text):
    return f"{code}{text}{_Ansi.RESET}" if _use_color() else text


def _highlight_urls(text):
    # Make any http(s) URL stand out so it's obvious where to go. Trailing
    # punctuation (comma, period, paren, quote) is kept out of the link.
    return re.sub(
        r"(https?://[^\s]+?)([.,;:)\]'\"]*)(?=\s|$)",
        lambda m: _c(_Ansi.CYAN + _Ansi.UNDERLINE, m.group(1)) + m.group(2),
        text,
    )


def print_error_box(title, body, accent=_Ansi.RED):
    """Print a framed, colored error block so failures (and what to do about
    them) are easy to spot in a long download log."""
    bar = "─" * 72
    print()
    print(_c(accent + _Ansi.BOLD, f"✗  {title}"))
    print(_c(accent, bar))
    print(_highlight_urls(body))
    print(_c(accent, bar))
    print()


def print_banner(config_path):
    """Print the ScanNet++ ASCII logo from the top of the config file."""
    try:
        with open(config_path) as f:
            lines = f.read().splitlines()
    except Exception:
        return
    if not lines or not lines[0].startswith("#"):
        return
    # Grab the first '#'-bordered box at the very top (the logo), stopping at
    # its closing border (the second all-'#' row).
    banner = [lines[0]]
    for ln in lines[1:]:
        if not ln.startswith("#"):
            break
        banner.append(ln)
        if set(ln.strip()) == {"#"} and len(ln.strip()) > 20:
            break
    print(_c(_Ansi.CYAN + _Ansi.BOLD, "\n".join(banner)))
    print()


def read_txt_list(path):
    with open(path) as f:
        lines = f.read().splitlines()

    return lines


def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j


def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)


def check_remote_file_exists(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: "HEAD"

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_scannetpp_gs(cfg, scene_ids):
    '''
    Download ScanNet++GS data
    '''

    print('Downloading ScanNet++GS data...')
    for scene_id in tqdm(scene_ids, desc="scenes"):
        src_path = Path('scannetpp_gs') / scene_id / 'ckpts' / 'point_cloud_30000.ply'
        tgt_path = Path(cfg.scannetpp_gs_dir) / f"{scene_id}/point_cloud_30000.ply"
        if not check_download_file(cfg, cfg.scannetpp_gs_url, src_path, tgt_path, cfg.dry_run):
            break


def urlretrieve_multi_trials(url, filename, max_trials=5):
    # 401 -> invalid token or token expired
    # 404 -> File not found
    # 406 -> Not acceptable, please update the download script
    for i in range(max_trials):
        try:
            urlretrieve(url, filename)
            time.sleep(0.2)     # wait a bit to prevent being rejected for frequent access to the server
            return True

        except urllib.error.ContentTooShortError as e:
            # Failed to download the file completely. Delete the file and retry.
            # Delete filename
            print("ERROR: Content too short. It is likely that the download was incomplete due to network issues. Retrying...")
            if i < max_trials - 1:
                if Path(filename).exists():
                    os.remove(filename)
                time.sleep(0.5)  # wait a bit before retrying
            else:
                print(f"Failed to download {url} after {max_trials} trials")
                raise e

        # HTTP error
        except urllib.error.HTTPError as e:
            if e.code == 401:
                # The server's response body explains whether the token is
                # invalid or expired, and (if expired) how to request an extension.
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    "Download failed — invalid or expired token",
                    server_msg or "It could be that an invalid or expired token is used.",
                )
                # Stop cleanly: SystemExit prints no traceback, so the message
                # above stays visible instead of being buried under a stack trace.
                raise SystemExit(1)

            elif e.code == 404:
                print_error_box(
                    "Download failed — file not found (404)",
                    f"{url}\n\n"
                    "This asset may not be available for this scene/split.",
                    accent=_Ansi.YELLOW,
                )
                raise SystemExit(1)

            elif e.code == 406:
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    "Download failed — download script out of date (406)",
                    (server_msg or "This download script is out of date.")
                    + "\n\nPlease download the latest script from the ScanNet++ "
                    "website and try again.",
                    accent=_Ansi.YELLOW,
                )
                raise SystemExit(1)
            elif e.code == 429 or 500 <= e.code < 600:
                # Transient server-side errors. Examples seen in the wild:
                # nginx 500 on NFS-backed small files (sporadic, ~7%) and
                # 429 from the per-IP rate limiter when many files land in
                # the same second. Back off and retry rather than abort
                # the whole download for a single bad request.
                if i < max_trials - 1:
                    delay = (0.5 * (2 ** i))   # 0.5s, 1s, 2s, 4s, 8s
                    print(f"WARN: HTTP {e.code} on {url} — retry {i+1}/{max_trials-1} in {delay:.1f}s")
                    if Path(filename).exists():
                        os.remove(filename)
                    time.sleep(delay)
                    continue
                # Fall through: out of retries.
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    f"Download failed — server returned HTTP {e.code} after {max_trials} tries",
                    (server_msg + "\n\n" if server_msg else "") + f"URL: {url}",
                )
                raise e
            else:
                try:
                    server_msg = e.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_msg = ""
                print_error_box(
                    f"Download failed — server returned HTTP {e.code}",
                    (server_msg + "\n\n" if server_msg else "") + f"URL: {url}",
                )
                raise e
    return False


def download_file(url, filename, verbose=True, make_parent=False):
    """
    Download file from url to filename
    """
    # download_url = url.

    if make_parent:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"{url} ==> {filename}")
    return urlretrieve_multi_trials(url, filename)


def check_download_file(cfg, url_template, remote_path, local_path, dry_run):
    """
    check if file exists, else download
    remote_path: relative to root
    local_path: full path
    dry_run: only check if file exists, don't download
    """
    # Updates: fix Windows users that use backslash for remote_path
    remote_path = str(remote_path).replace("\\", "/")
    url = url_template.replace("TOKEN", cfg.token).replace("FILEPATH", remote_path)

    if dry_run:
        status = check_remote_file_exists(url)
        if status:
            print("Remote file exists:", url)
        else:
            print("Remote file missing:", url)
        return status

    if local_path.is_file():
        if cfg.verbose:
            print("File exists, skipping download: ", local_path)
        return True
    else:
        return download_file(url, local_path, verbose=cfg.verbose, make_parent=True)


def read_env_token():
    """Return SCANNETPP_TOKEN from the environment, or from a .env file in
    the current folder, this script's folder, or the repo root."""
    if os.environ.get("SCANNETPP_TOKEN"):
        return os.environ["SCANNETPP_TOKEN"].strip()
    script_dir = Path(__file__).resolve().parent
    for env_file in [Path.cwd() / ".env", script_dir / ".env", script_dir.parent / ".env"]:
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "SCANNETPP_TOKEN":
                return value.strip().strip("\"'")
    return None


def main(args):
    print_banner(args.config_file)
    cfg = load_yaml_munch(args.config_file)
    # Token from the SCANNETPP_TOKEN env var or the repo's .env takes priority over the yml
    env_token = read_env_token()
    if env_token:
        cfg.token = env_token
        print("Using token from SCANNETPP_TOKEN (.env or environment)")
    # Ask the user to provide the token if not provided
    elif cfg.get("token", "<YOUR_TOKEN_HERE>") == "<YOUR_TOKEN_HERE>":
        cfg.token = input("Please enter your download token: ").strip()
        if cfg.token == "":
            print("No token provided, exiting. Please apply for a token from ScanNet++ official website.")
            return
    else:
        print(f"Using token from config file: {cfg.token}")

    if cfg.get("data_root", "<DOWNLOAD_LOCATION_HERE>") == "<DOWNLOAD_LOCATION_HERE>":
        cfg.data_root = input("Please enter your download location: (default: ./scannetpp_data)").strip()
        if cfg.data_root == "":
            cfg.data_root = "./scannetpp_data"
    print(f"Downloading to: {cfg.data_root}")

    # Notify user the size of the whole dataset
    print(
        "WARNING: Downloading the full ScanNet++ dataset with default splits and assets "
        "will require approximately 1.5TB of disk space.\n"
        "If you have already customized splits or assets in your config file, the required space may differ.\n"
        "Do you want to proceed with the download? (y/n)\n > ", end=""
    )
    ans = input().strip().lower()
    if ans != 'y':
        print("Exiting.")
        return

    if cfg.dry_run:
        print("Dry run: check if remote files exist, no files will be downloaded")

    missing = []

    data_root = Path(cfg.data_root)

    # create data root directory
    data_root.mkdir(parents=True, exist_ok=True)

    # download meta files
    for path in cfg.meta_files:
        if not check_download_file(cfg, cfg.root_url, path, data_root / path, cfg.dry_run):
            missing.append(str(data_root / path))

    if cfg.metadata_only:
        print("Downloaded metadata, done.")
        return

    # read all the split files
    split_lists = {}
    for split in cfg.splits:
        split_path = data_root / "splits" / f"{split}.txt"
        split_lists[split] = read_txt_list(split_path)

    # get the list of scenes to be downloaded
    if cfg.get("download_scenes"):
        scene_ids = cfg.download_scenes
    elif cfg.get("download_splits"):
        scene_ids = []
        for split in cfg.download_splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    # we know the scene ids, check for 3rd party datasets
    if cfg.get("scannetpp_gs_dir"):
        download_scannetpp_gs(cfg, scene_ids)
        print(f"Downloaded ScanNet++GS data to {cfg.scannetpp_gs_dir}, done.")
        return

    # get the list of assets to download for these scenes
    if cfg.get("download_assets"):
        download_assets = cfg.download_assets
    elif cfg.get("download_options"):
        download_assets = []
        for option in cfg.download_options:
            option_assets = cfg.option_assets[option]
            for asset in option_assets:
                if asset not in download_assets:
                    download_assets.append(asset)
    else:
        download_assets = cfg.default_assets

    print("Downloading assets:", download_assets)
    print("Scenes selected: ", len(scene_ids))

    download_has_error = False
    for scene_id in tqdm(scene_ids, desc="scenes"):
        # download from here
        # path relative to root url
        src_scene = ScannetppScene_Release(scene_id, data_root="data")
        # to here
        tgt_scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")

        # get the split for this scene
        split = None
        for split in cfg.splits:
            if scene_id in split_lists[split]:
                break
        assert split is not None, f"Scene {scene_id} not in any split"

        for asset in tqdm(download_assets, desc="assets", leave=False):
            # some assets not present in test splits
            if asset in cfg.exclude_assets.get(split, []):
                continue

            # check if asset is zipped, download the zip and unzip it to the target path
            if asset in cfg.zipped_assets:
                tgt_path = getattr(tgt_scene, asset)

                if tgt_path.is_file() or tgt_path.is_dir():
                    if cfg.verbose:
                        print("File exists, skipping download: ", tgt_path)
                    continue

                src_download_path = getattr(src_scene, asset).with_suffix(".zip")
                tgt_download_path = tgt_path.with_suffix(".zip")

                if not check_download_file(cfg, cfg.root_url, src_download_path, tgt_download_path, cfg.dry_run):
                    missing.append(str(tgt_download_path))
                    # Abort the downloading process
                    download_has_error = True
                    break

                if not cfg.dry_run:
                    # unzip it
                    if cfg.verbose:
                        print("Unzipping:", tgt_download_path)
                    with zipfile.ZipFile(tgt_download_path, "r") as zip_ref:
                        zip_ref.extractall(tgt_download_path.parent)
                    # remove the zip file
                    if cfg.verbose:
                        print("Delete zip file:", tgt_download_path)
                    tgt_download_path.unlink()
            else:
                #  download single file
                src_path = getattr(src_scene, asset)
                tgt_path = getattr(tgt_scene, asset)
                if not check_download_file(cfg, cfg.root_url, src_path, tgt_path, cfg.dry_run):
                    missing.append(str(tgt_path))
                    download_has_error = True
                    break

        if download_has_error:
            print(f"\nError downloading scene {scene_id}, aborting.")
            break

    if missing:
        print(f"{len(missing)} files missing:", missing)
    else:
        print("Download successful!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
