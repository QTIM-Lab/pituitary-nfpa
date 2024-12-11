#!/usr/bin/env bash¬

SIF_FILE="/autofs/space/crater_001/projects/pituitary-nfpa/singularity.sif"
singularity shell \
    --nv \
    --writable-tmpfs \
    -B /autofs/:/autofs/ \
    $SIF_FILE
