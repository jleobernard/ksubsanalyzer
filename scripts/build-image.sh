#!/usr/bin/env bash
set -e
KSA_VERSION=0.0.1

FULL_PATH_TO_SCRIPT="$(realpath "$0")"
SCRIPT_DIRECTORY="$(dirname "$FULL_PATH_TO_SCRIPT")"
PROJECT_BASE="$(realpath "$SCRIPT_DIRECTORY/..")"
cd $PROJECT_BASE
docker build $PROJECT_BASE -t jleobernard/ksubsanalyzer:$KSA_VERSION
docker login
docker push jleobernard/ksubsanalyzer:$KSA_VERSION
#docker logout