# Downloading a Google Drive folder with rclone

For datasets and checkpoints. `gdown --folder` silently truncates at 50 files
per folder, so use rclone for anything real.

## 1. Install

```bash
brew install rclone                                    # macOS
curl https://rclone.org/install.sh | sudo bash         # Linux
```

## 2. Configure

```bash
rclone config
```

| Prompt | Answer |
|---|---|
| `e/n/d/r/c/s/q>` | `n` |
| `name>` | `gdrive` |
| `Storage>` | `drive` (type the word; numbers shift between versions) |
| `client_id>` / `client_secret>` | *(blank)* |
| `scope>` | `2` (read-only) |
| `service_account_file>` | *(blank)* |
| `Edit advanced config?` | `n` |
| `Use auto config?` | `y` local, `n` headless |
| `Configure this as a Shared Drive?` | `n` |
| `Keep this remote?` | `y`, then `q` |

## 3. Get the folder ID

Last path segment of the share link:

```
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz?usp=sharing
                                       └──────────── ID ──────────┘
```

## 4. Verify, then copy

List first — confirms auth and access without pulling gigabytes:

```bash
FID=1AbCdEfGhIjKlMnOpQrStUvWxYz
rclone lsd  --drive-root-folder-id $FID gdrive:
rclone size --drive-root-folder-id $FID gdrive:
```

```bash
rclone copy --drive-root-folder-id $FID gdrive: ./dest \
    -P --transfers 8 --checkers 16
```

`copy` never deletes at the destination (unlike `sync`), and it is resumable —
re-run the same command to skip what already arrived.

## Headless box

Easiest: configure on the laptop, then ship the config.

```bash
scp ~/.config/rclone/rclone.conf user@gpu-box:~/.config/rclone/rclone.conf
```

Native alternative: answer `n` to *Use auto config?*. rclone prints a command
like `rclone authorize "drive" "eyJzY29..."` — run that exact string on a
machine with a browser, then paste the returned token back.

## "Shared with me" instead of a folder ID

```bash
rclone lsd  gdrive: --drive-shared-with-me
rclone copy gdrive:"Folder Name" ./dest --drive-shared-with-me -P
```

Or add a shortcut to your own Drive in the web UI, after which plain
`gdrive:path/to/folder` works.

## Flags worth knowing

| Flag | When |
|---|---|
| `--tpslimit 10` | `403 rateLimitExceeded` on folders with many small files |
| `--drive-acknowledge-abuse` | Google flagged a large archive it could not virus-scan |
| `--exclude "*.DS_Store"` | Always, when the source was ever touched by macOS |
| `--dry-run` | Preview before a large transfer |
