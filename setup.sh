#!/usr/bin/env bash

set -eu

uv --version || wget -qO- https://astral.sh/uv/install.sh | sh

uv sync

mkdir -p input model

echo "Do u wanna leave the .zip files [y/n]"
read -r conf
while [[ "$conf" != "y" && "$conf" != "n" ]]; do # FIXED: added [[ ]]
  echo "just [y/n] don't waste time"
  read -r conf
done

(
  cd input

  wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz

  file=empatheticdialogues.tar.gz
  if [ -f "$file" ]; then
    tar -xzf "$file" || echo "$file is corrupted"
    [ "$conf" == "n" ] && rm "$file"
  fi
)
echo "Download complete!"
