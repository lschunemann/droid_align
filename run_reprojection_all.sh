#!/bin/bash
# =============================================================================
# Run mesh reprojection for all camera directories in droid_processed
#
# Only processes cameras that have:
#   - intrinsics.json (required)
#   - objects/ directory with meshes (required)
#   - Optionally: extrinsics.json (for filtering, use --require-extrinsics)
#
# Usage:
#   ./run_reprojection_all.sh /path/to/droid_processed
#   ./run_reprojection_all.sh /path/to/droid_processed --require-extrinsics
#   ./run_reprojection_all.sh /path/to/droid_processed --mode wireframe
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPROJECT_SCRIPT="${SCRIPT_DIR}/reproject_mesh.py"

# Default values
INPUT_DIR=""
MODE="bbox"
REQUIRE_EXTRINSICS=0
DEBUG=0
FORCE_STEREO=0
USE_FILTER=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --require-extrinsics)
            REQUIRE_EXTRINSICS=1
            shift
            ;;
        --force-stereo)
            FORCE_STEREO=1
            shift
            ;;
        --filter)
            USE_FILTER=1
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <droid_processed_dir> [options]"
            echo ""
            echo "Options:"
            echo "  --mode MODE           Visualization mode: bbox, wireframe, vertices (default: bbox)"
            echo "  --require-extrinsics  Only process cameras that have extrinsics.json"
            echo "  --force-stereo        Re-process stereo reprojections (delete existing and redo)"
            echo "  --filter              Only reproject objects listed in visible_objects.json"
            echo "  --debug               Print debug information"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory required"
    echo "Usage: $0 <droid_processed_dir> [options]"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Directory not found: $INPUT_DIR"
    exit 1
fi

if [[ ! -f "$REPROJECT_SCRIPT" ]]; then
    echo "ERROR: reproject_mesh.py not found at: $REPROJECT_SCRIPT"
    exit 1
fi

echo "============================================="
echo "Mesh Reprojection - Batch Processing"
echo "============================================="
echo "Input directory: $INPUT_DIR"
echo "Mode: $MODE"
echo "Require extrinsics: $REQUIRE_EXTRINSICS"
echo "Use filter: $USE_FILTER"
echo "============================================="
echo ""

# Counters
TOTAL=0
PROCESSED=0
SKIPPED_NO_INTRINSICS=0
SKIPPED_NO_OBJECTS=0
SKIPPED_NO_EXTRINSICS=0
ERRORS=0

# Build debug flag
DEBUG_FLAG=""
if [[ $DEBUG -eq 1 ]]; then
    DEBUG_FLAG="--debug"
fi

# Build filter flag
FILTER_FLAG=""
if [[ $USE_FILTER -eq 1 ]]; then
    FILTER_FLAG="--filter"
fi

# Find all camera directories
# Structure: droid_processed/env/outcome/timestamp/camera_serial/
for env_dir in "$INPUT_DIR"/*/; do
    env_name=$(basename "$env_dir")

    # Skip non-directories and special folders
    [[ ! -d "$env_dir" ]] && continue
    [[ "$env_name" == "calibration" ]] && continue

    for outcome_dir in "$env_dir"/*/; do
        outcome=$(basename "$outcome_dir")

        # Only process success/failure directories
        [[ ! -d "$outcome_dir" ]] && continue
        [[ "$outcome" != "success" && "$outcome" != "failure" ]] && continue

        for traj_dir in "$outcome_dir"/*/; do
            [[ ! -d "$traj_dir" ]] && continue
            traj_name=$(basename "$traj_dir")

            # Process regular camera directories
            for cam_dir in "$traj_dir"/*/; do
                [[ ! -d "$cam_dir" ]] && continue
                cam_name=$(basename "$cam_dir")

                # Skip special directories
                [[ "$cam_name" == "stereo" ]] && continue
                [[ "$cam_name" == "depth" ]] && continue
                [[ "$cam_name" == "objects" ]] && continue

                TOTAL=$((TOTAL + 1))
                rel_path="${env_name}/${outcome}/${traj_name}/${cam_name}"

                # Check for intrinsics
                if [[ ! -f "$cam_dir/intrinsics.json" ]]; then
                    if [[ $DEBUG -eq 1 ]]; then
                        echo "[SKIP] $rel_path - no intrinsics.json"
                    fi
                    SKIPPED_NO_INTRINSICS=$((SKIPPED_NO_INTRINSICS + 1))
                    continue
                fi

                # Check for objects directory
                if [[ ! -d "$cam_dir/objects" ]]; then
                    if [[ $DEBUG -eq 1 ]]; then
                        echo "[SKIP] $rel_path - no objects/"
                    fi
                    SKIPPED_NO_OBJECTS=$((SKIPPED_NO_OBJECTS + 1))
                    continue
                fi

                # Check for extrinsics if required
                if [[ $REQUIRE_EXTRINSICS -eq 1 && ! -f "$cam_dir/extrinsics.json" ]]; then
                    if [[ $DEBUG -eq 1 ]]; then
                        echo "[SKIP] $rel_path - no extrinsics.json"
                    fi
                    SKIPPED_NO_EXTRINSICS=$((SKIPPED_NO_EXTRINSICS + 1))
                    continue
                fi

                # Skip if already processed
                if [[ -f "$cam_dir/reprojection_${MODE}.png" ]]; then
                    if [[ $DEBUG -eq 1 ]]; then
                        echo "[SKIP] $rel_path - already processed"
                    fi
                    PROCESSED=$((PROCESSED + 1))
                    continue
                fi

                # Run reprojection
                echo "Processing: $rel_path"
                if python3 "$REPROJECT_SCRIPT" "$cam_dir" --mode "$MODE" $FILTER_FLAG $DEBUG_FLAG; then
                    PROCESSED=$((PROCESSED + 1))
                else
                    echo "[ERROR] Failed: $rel_path"
                    ERRORS=$((ERRORS + 1))
                fi
            done

            # Process stereo directories
            # Structure: traj/stereo/{cam_id}-stereo/{side}_{frame}/objects/
            stereo_base="$traj_dir/stereo"
            if [[ -d "$stereo_base" ]]; then
                for stereo_cam_dir in "$stereo_base"/*/; do
                    [[ ! -d "$stereo_cam_dir" ]] && continue
                    stereo_cam_name=$(basename "$stereo_cam_dir")

                    for frame_dir in "$stereo_cam_dir"/left_*/; do
                        [[ ! -d "$frame_dir" ]] && continue
                        frame_name=$(basename "$frame_dir")

                        # Find the frame image and intrinsics once (shared by both variants)
                        image_path="${stereo_cam_dir}/${frame_name}.png"
                        frame_idx="${frame_name#*_}"
                        intrinsics_path="${traj_dir}/depth/${stereo_cam_name}/frame_${frame_idx}/intrinsics.npy"
                        if [[ ! -f "$intrinsics_path" ]]; then
                            intrinsics_path="${traj_dir}/depth/frame_${frame_idx}/intrinsics.npy"
                        fi

                        # Process both objects/ (no pointmap) and objects_pointmap/
                        for variant in "objects" "objects_pointmap"; do
                            if [[ "$variant" == "objects" ]]; then
                                pointmap_flag=""
                                output_file="reprojection_${MODE}.png"
                            else
                                pointmap_flag="--pointmap"
                                output_file="reprojection_pointmap_${MODE}.png"
                            fi

                            # Only process if the objects dir exists
                            [[ ! -d "$frame_dir/$variant" ]] && continue

                            TOTAL=$((TOTAL + 1))
                            rel_path="${env_name}/${outcome}/${traj_name}/stereo/${stereo_cam_name}/${frame_name}/${variant}"

                            # Skip if already processed (unless --force-stereo)
                            if [[ -f "$frame_dir/$output_file" ]]; then
                                if [[ $FORCE_STEREO -eq 1 ]]; then
                                    rm -f "$frame_dir/$output_file"
                                    if [[ $DEBUG -eq 1 ]]; then
                                        echo "[REDO] $rel_path"
                                    fi
                                else
                                    if [[ $DEBUG -eq 1 ]]; then
                                        echo "[SKIP] $rel_path - already processed"
                                    fi
                                    PROCESSED=$((PROCESSED + 1))
                                    continue
                                fi
                            fi

                            if [[ ! -f "$image_path" ]]; then
                                if [[ $DEBUG -eq 1 ]]; then
                                    echo "[SKIP] $rel_path - no image ${frame_name}.png"
                                fi
                                SKIPPED_NO_INTRINSICS=$((SKIPPED_NO_INTRINSICS + 1))
                                continue
                            fi

                            if [[ ! -f "$intrinsics_path" ]]; then
                                if [[ $DEBUG -eq 1 ]]; then
                                    echo "[SKIP] $rel_path - no intrinsics"
                                fi
                                SKIPPED_NO_INTRINSICS=$((SKIPPED_NO_INTRINSICS + 1))
                                continue
                            fi

                            # Run reprojection
                            echo "Processing: $rel_path"
                            if python3 "$REPROJECT_SCRIPT" "$frame_dir" --mode "$MODE" --intrinsics "$intrinsics_path" --image "$image_path" $pointmap_flag $FILTER_FLAG $DEBUG_FLAG; then
                                PROCESSED=$((PROCESSED + 1))
                            else
                                echo "[ERROR] Failed: $rel_path"
                                ERRORS=$((ERRORS + 1))
                            fi
                        done
                    done
                done
            fi
        done
    done
done

echo ""
echo "============================================="
echo "Summary"
echo "============================================="
echo "Total camera directories found: $TOTAL"
echo "Successfully processed: $PROCESSED"
echo "Skipped (no intrinsics): $SKIPPED_NO_INTRINSICS"
echo "Skipped (no objects): $SKIPPED_NO_OBJECTS"
if [[ $REQUIRE_EXTRINSICS -eq 1 ]]; then
    echo "Skipped (no extrinsics): $SKIPPED_NO_EXTRINSICS"
fi
echo "Errors: $ERRORS"
echo "============================================="

if [[ $PROCESSED -gt 0 ]]; then
    echo ""
    echo "Output files: <camera_dir>/reprojection.png"
fi
