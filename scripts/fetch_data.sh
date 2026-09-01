#!/usr/bin/env bash
#
# Fetch the dataset archives from Google Drive, extract them in place, and
# delete the archives.
#
# The Drive folder mirrors the layout of ./data, with audio.zip / codecs.zip per
# dataset (Emilia is split across numbered shards). Each archive is extracted
# into its own directory and removed only after unzip reports success, so the
# script is safe to re-run: rclone skips files already transferred and the
# extract pass finds nothing left to do.
#
#   ./scripts/fetch_data.sh                     # download + extract into ./data
#   ./scripts/fetch_data.sh --dry-run           # show what would happen
#   ./scripts/fetch_data.sh --skip-download     # extract archives already present
#   ./scripts/fetch_data.sh --dest /mnt/data    # somewhere else

set -euo pipefail

FOLDER_ID="1Fz9t1dniEjYu5FUsxPD5gZF6hWscEk2k"
REMOTE="gdrive:"
DEST="data"
TRANSFERS=8
CHECKERS=16
DO_DOWNLOAD=1
DO_EXTRACT=1
KEEP_ZIPS=0
DRY_RUN=0
IMPORT_CONFIG=""
AUTO_CONFIG=1
FORCE_HEADLESS=0
FORCE_BROWSER=0

usage() {
    sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --folder-id ID    Google Drive folder ID (default: the OmniVoice dataset)
  --remote NAME     rclone remote, including the colon (default: gdrive:)
  --dest DIR        destination directory (default: data)
  --transfers N     parallel rclone transfers (default: 8)
  --import-config F install an existing rclone.conf (e.g. scp'd from a laptop)
  --headless        force the paste-a-token auth flow (no browser on this host)
  --browser         force the open-a-browser auth flow
  --no-config       fail instead of offering to configure a missing remote
  --skip-download   do not touch Drive; just extract what is already on disk
  --skip-extract    download only, leave the .zip files alone
  --keep-zips       extract but do not delete the archives
  --dry-run         print the plan without downloading, extracting or deleting
  -h, --help        this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --folder-id)  FOLDER_ID="$2"; shift 2 ;;
        --remote)     REMOTE="$2"; shift 2 ;;
        --dest)       DEST="$2"; shift 2 ;;
        --transfers)  TRANSFERS="$2"; shift 2 ;;
        --import-config) IMPORT_CONFIG="$2"; shift 2 ;;
        --headless)   FORCE_HEADLESS=1; shift ;;
        --browser)    FORCE_BROWSER=1; shift ;;
        --no-config)  AUTO_CONFIG=0; shift ;;
        --skip-download) DO_DOWNLOAD=0; shift ;;
        --skip-extract)  DO_EXTRACT=0; shift ;;
        --keep-zips)  KEEP_ZIPS=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# Written as `if` rather than `[ ... ] && ...`: under `set -e` an AND-OR list
# whose first test is false returns 1 and silently kills the script.
if [ "$DO_DOWNLOAD" -eq 1 ] && ! have rclone; then
    die "rclone not found (brew install rclone)"
fi
if [ "$DO_EXTRACT" -eq 1 ] && ! have unzip; then
    die "unzip not found"
fi

human() {  # bytes -> human readable, without depending on numfmt
    awk -v b="$1" 'BEGIN{
        split("B KiB MiB GiB TiB", u, " "); i = 1
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf (i == 1 ? "%d %s" : "%.1f %s"), b, u[i]
    }'
}

# ------------------------------------------------------------ remote setup ---
remote_exists() {
    rclone listremotes 2>/dev/null | grep -qx -- "${1}:"
}

is_headless() {
    if [ "$FORCE_HEADLESS" -eq 1 ]; then return 0; fi
    if [ "$FORCE_BROWSER" -eq 1 ]; then return 1; fi
    # A browser-based OAuth callback needs a reachable 127.0.0.1:53682, which a
    # remote shell does not have. macOS desktop and X11 sessions do.
    if [ "$(uname -s)" = "Darwin" ] && [ -z "${SSH_CONNECTION:-}" ]; then return 1; fi
    if [ -n "${DISPLAY:-}" ]; then return 1; fi
    return 0
}

ensure_remote() {
    local name="${REMOTE%:}"

    if [ -n "$IMPORT_CONFIG" ]; then
        [ -f "$IMPORT_CONFIG" ] || die "no such file: $IMPORT_CONFIG"
        local target
        target=$(rclone config file 2>/dev/null | tail -1)
        [ -n "$target" ] || target="$HOME/.config/rclone/rclone.conf"
        echo "==> installing rclone config -> $target"
        if [ "$DRY_RUN" -eq 0 ]; then
            mkdir -p "$(dirname "$target")"
            cp "$IMPORT_CONFIG" "$target"
            chmod 600 "$target"
        fi
    fi

    if remote_exists "$name"; then
        return 0
    fi

    if [ "$AUTO_CONFIG" -eq 0 ]; then
        die "rclone remote '${name}:' is not configured (--no-config was given)"
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "==> would configure rclone remote '${name}:' (drive, read-only)"
        return 0
    fi

    echo "==> rclone remote '${name}:' not found; configuring it now"
    echo "    type: drive, scope: drive.readonly (read-only access)"
    if is_headless; then
        cat <<EOF

    This host has no browser, so rclone will print a command that looks like

        rclone authorize "drive" "eyJzY29..."

    Run that ENTIRE line on a machine that has a browser (your laptop),
    complete the Google sign-in, then paste the token it prints back here.

EOF
        rclone config create "$name" drive scope=drive.readonly config_is_local=false
    else
        echo "    A browser window will open for Google sign-in."
        echo
        rclone config create "$name" drive scope=drive.readonly
    fi

    remote_exists "$name" || die "remote '${name}:' still not configured; aborting"
    echo "==> remote '${name}:' configured"
    echo "    note: this stores a long-lived OAuth token in $(rclone config file 2>/dev/null | tail -1)."
    echo "    revoke later at https://myaccount.google.com/permissions if this host is not yours."
}

check_folder() {
    if rclone lsd --drive-root-folder-id "$FOLDER_ID" "$REMOTE" >/dev/null 2>&1; then
        return 0
    fi
    echo "error: cannot list folder '$FOLDER_ID' on '$REMOTE'" >&2
    if [ "${#FOLDER_ID}" -ne 33 ]; then
        echo "  the ID is ${#FOLDER_ID} characters; Drive folder IDs are 33 and start with '1'." >&2
        echo "  a truncated copy/paste is the usual cause of a 404 here." >&2
    fi
    echo "  check that the Google account you authorized can open the share link," >&2
    echo "  or try: rclone lsd $REMOTE --drive-shared-with-me" >&2
    exit 1
}

# ---------------------------------------------------------------- download ---
if [ "$DO_DOWNLOAD" -eq 1 ]; then
    ensure_remote
    if [ "$DRY_RUN" -eq 0 ]; then
        check_folder
    fi
    echo "==> downloading ${REMOTE}(folder ${FOLDER_ID}) -> ${DEST}/"
    RCLONE_ARGS=(
        copy --drive-root-folder-id "$FOLDER_ID" "$REMOTE" "$DEST"
        --transfers "$TRANSFERS" --checkers "$CHECKERS"
        --exclude "._*" --exclude ".DS_Store" --exclude "__MACOSX/**"
    )
    if [ "$DRY_RUN" -eq 1 ]; then
        if remote_exists "${REMOTE%:}"; then
            rclone "${RCLONE_ARGS[@]}" --dry-run
        else
            echo "    (transfer preview skipped: remote not configured yet)"
        fi
    else
        mkdir -p "$DEST"
        rclone "${RCLONE_ARGS[@]}" -P
    fi
fi

# ----------------------------------------------------------------- extract ---
if [ "$DO_EXTRACT" -eq 1 ]; then
    [ -d "$DEST" ] || die "destination '$DEST' does not exist"
    echo "==> extracting archives under ${DEST}/"

    total=0
    failed=0
    bytes=0
    while IFS= read -r -d '' zip; do
        total=$((total + 1))
        dir=$(dirname "$zip")
        size=$(wc -c < "$zip" | tr -d ' ')
        rel="${dir#"$DEST"}"; rel="${rel#/}"; rel="${rel:-.}"
        printf '  [%2d] %-40s %10s -> %s/\n' \
            "$total" "$(basename "$zip")" "$(human "$size")" "$rel"

        if [ "$DRY_RUN" -eq 1 ]; then
            continue
        fi

        # -o overwrites, which makes a re-run after an interrupted extract safe.
        # __MACOSX and ._* are AppleDouble junk from zips created on macOS.
        if unzip -q -o "$zip" -x '__MACOSX/*' '._*' -d "$dir"; then
            bytes=$((bytes + size))
            if [ "$KEEP_ZIPS" -eq 0 ]; then
                rm -f "$zip"
            fi
        else
            # Leave the archive in place so a re-run can retry it.
            echo "      FAILED to extract, archive kept" >&2
            failed=$((failed + 1))
        fi
    done < <(find "$DEST" -type f -name '*.zip' -print0 | sort -z)

    if [ "$total" -eq 0 ]; then
        echo "  no .zip files found -- nothing to do"
    elif [ "$DRY_RUN" -eq 0 ]; then
        echo "==> tidying macOS metadata"
        find "$DEST" -type d -name '__MACOSX' -prune -exec rm -rf {} + 2>/dev/null || true
        find "$DEST" -type f \( -name '._*' -o -name '.DS_Store' \) -delete 2>/dev/null || true

        kept=""
        if [ "$KEEP_ZIPS" -eq 1 ]; then kept=" (archives kept)"; fi
        echo "==> extracted $((total - failed))/${total} archives, $(human "$bytes") of zips${kept}"
        if [ "$failed" -gt 0 ]; then
            die "$failed archive(s) failed; re-run to retry"
        fi
    fi
fi

echo "==> done"
