# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/utils/env.py

"""Set up Environment."""

_ENV_SETUP_DONE = False


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
