# Badger_detection.py

# %% CELL 1: Imports and Configuration
import os
import cv2 # OpenCV for video processing
import subprocess
import json
import tempfile
import shutil
import argparse

# --- Configuration (MODIFY THESE PATHS AND SETTINGS) ---
DEFAULT_VIDEO_PATH = "3_IMG_0019.MP4" # Placeholder: Replace with your video path
BASE_OUTPUT_DIR = "output" # Directory to save frames and results

# SpeciesNet settings
FRAME_INTERVAL_SECONDS = 0  # Extract one frame every N seconds (e.g., 0.2 for 5fps processing, 0 to process all frames)
COUNTRY_CODE = "GBR"  # Optional: e.g., "GBR" for Great Britain. Set to None to disable.
# ADMIN1_REGION = None # Optional: e.g., "CA" for California if country is "USA"

# Ensure the base output directory exists
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# %% CELL 2: Frame Extraction Function
def extract_frames(video_path, output_folder, interval_seconds=1):
    """
    Extracts frames from a video file at a given interval.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save extracted frames.
        interval_seconds (int): Interval in seconds between extracted frames.

    Returns:
        tuple: (list of extracted frame paths, original video FPS) or ([], None) on error.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return [], None

    os.makedirs(output_folder, exist_ok=True)

    # Get video properties first
    cap_props = cv2.VideoCapture(video_path)
    if not cap_props.isOpened():
        print(f"Error: Could not open video file {video_path} to get properties.")
        return [], None
    
    original_fps = cap_props.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap_props.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_props.release()

    if original_fps is None or original_fps <= 0:
        print(f"Error: Invalid FPS ({original_fps}) obtained from video {video_path}.")
        return [], None
    if total_video_frames <= 0:
        print(f"Error: Invalid total frame count ({total_video_frames}) from video {video_path}.")
        return [], None

    # Calculate the actual stride between frames to be saved
    # If interval_seconds is 0, process every frame (stride = 1)
    # Otherwise, stride is fps * interval_seconds (but at least 1)
    effective_extraction_stride = 1
    if interval_seconds > 0:
        effective_extraction_stride = max(1, int(original_fps * interval_seconds))
    
    # Calculate how many frames we expect to save
    # Using ceiling division: (numerator + denominator - 1) // denominator
    expected_saved_frame_count = (total_video_frames + effective_extraction_stride - 1) // effective_extraction_stride
    
    print(f"Video properties: Total Frames={total_video_frames}, FPS={original_fps:.2f}, IntervalSec={interval_seconds}, Stride={effective_extraction_stride}, Expected Saved Frames={expected_saved_frame_count}")

    # Check if frames already exist
    existing_frames_paths = sorted([
        os.path.join(output_folder, f) 
        for f in os.listdir(output_folder) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) # More general check
    ])

    if existing_frames_paths:
        # Allow a small tolerance (e.g. +/-1) due to potential minor variations in total_video_frames reporting
        if abs(len(existing_frames_paths) - expected_saved_frame_count) <= 1:
            print(f"Found {len(existing_frames_paths)} existing frames in {output_folder}, which matches/is close to the expected count of {expected_saved_frame_count}. Skipping extraction.")
            return existing_frames_paths, original_fps
        else:
            print(f"Found {len(existing_frames_paths)} existing frames, but expected {expected_saved_frame_count}. Re-extracting frames.")
            # Clear out the old frames to ensure a fresh extraction
            for old_frame in existing_frames_paths:
                try:
                    os.remove(old_frame)
                except OSError as e:
                    print(f"Warning: Could not delete old frame {old_frame}: {e}")
            print(f"Cleared old frames from {output_folder}.")
            existing_frames_paths = [] # Reset this list
    else:
        print(f"No existing frames found in {output_folder}. Proceeding with extraction.")
    
    # Proceed with frame extraction
    cap_extract = cv2.VideoCapture(video_path)
    if not cap_extract.isOpened():
        # This check is somewhat redundant due to cap_props, but good for safety
        print(f"Error: Could not open video file {video_path} for extraction (second attempt).")
        return [], None

    # Use effective_extraction_stride for the loop logic
    # The loop variable `frame_count` is the 0-indexed current frame number from the video.
    # We save a frame if `frame_count % effective_extraction_stride == 0`.

    extracted_frames_paths = []
    current_frame_number = 0 # Overall frame number in the video
    saved_frame_count_this_run = 0

    while True:
        ret, frame = cap_extract.read()
        if not ret:
            break 

        if current_frame_number % effective_extraction_stride == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count_this_run:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frames_paths.append(frame_filename)
            saved_frame_count_this_run += 1
        
        current_frame_number += 1

    cap_extract.release()
    
    if saved_frame_count_this_run != expected_saved_frame_count:
        print(f"Warning: Extracted {saved_frame_count_this_run} frames, but expected {expected_saved_frame_count}.")
        # This might happen if total_video_frames reported by cv2 is slightly off from actual readable frames.
        # The script will proceed with the frames it actually managed to save.

    print(f"Extracted {saved_frame_count_this_run} frames from {video_path} to {output_folder}")
    return extracted_frames_paths, original_fps

# %% CELL 3: Run SpeciesNet Function
def run_speciesnet_on_frames(frames_folder_path, output_json_path, country_code=None, admin1_region=None):
    """
    Runs SpeciesNet on a folder of images.

    Args:
        frames_folder_path (str): Path to the folder containing image frames.
        output_json_path (str): Path to save the SpeciesNet JSON output.
        country_code (str, optional): ISO 3166-1 alpha-3 country code.
        admin1_region (str, optional): First-level administrative division (e.g., US state).

    Returns:
        str: Path to the predictions JSON file if successful, None otherwise.
    """
    if not os.path.isdir(frames_folder_path):
        print(f"Error: Frames folder not found at {frames_folder_path}")
        return None
    
    if not os.listdir(frames_folder_path):
        print(f"Error: No frames found in {frames_folder_path}. Skipping SpeciesNet.")
        return None

    # Convert paths to absolute to avoid issues with cwd
    abs_frames_folder_path = os.path.abspath(frames_folder_path)
    abs_output_json_path = os.path.abspath(output_json_path)

    # Ensure the command uses the python from the activated virtual environment
    # This assumes the script is run after activating the speciesnet venv
    # python_executable = "python" # Or specify full path to venv python if needed
    python_executable = r"D:\Projects\Badger Detection\VScode\Badger_1.1\cameratrapai\.env\Scripts\python.exe"

    #cmd = [
    #    python_executable, "-m", "speciesnet.scripts.run_model",
    #    "--folders", abs_frames_folder_path,
    #    "--predictions_json", abs_output_json_path
    #]
    cmd = [
        "python -m speciesnet.scripts.run_model",
        "--folders", abs_frames_folder_path,
        "--predictions_json", abs_output_json_path
    ]

    if country_code:
        cmd.extend(["--country", country_code])
    if admin1_region and country_code == "USA": # admin1_region often specific to USA in examples
        cmd.extend(["--admin1_region", admin1_region])

    print("\n" + "="*70)
    print("STEP 2: RUNNING SPECIESNET (Object Detection & Classification)")
    print("="*70)
    print(f"  Input frames folder: {abs_frames_folder_path}")
    print(f"  Output JSON: {abs_output_json_path}")
    print(f"  SpeciesNet command: {' '.join(cmd)}")
    print("  This may take a significant amount of time depending on video length and hardware...")
    
    try:
        # It's important to run this from a directory where SpeciesNet can find its models,
        # or ensure models are downloaded. The script usually handles this.
        # The `cameratrapai` directory is a good candidate for `cwd` if issues arise.
        # However, `speciesnet` installed via pip should place scripts in PATH or make them callable.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="./cameratrapai") 
        
        try:
            stdout, stderr = process.communicate(timeout=5400) # 90 minutes timeout
        except subprocess.TimeoutExpired:
            process.kill()
            # Capture any final output after attempting to kill
            timeout_stdout, timeout_stderr = process.communicate()
            print("-" * 70)
            print("STEP 2: SPECIESNET TIMED OUT.")
            # Existing error details will follow
            print("-" * 70)
            return None

        if process.returncode == 0:
            print("-" * 70)
            print("STEP 2: SPECIESNET COMPLETED SUCCESSFULLY.")
            print(f"  Results saved to: {output_json_path}")
            print("-" * 70)
            if stderr: # SpeciesNet might output info to stderr even on success
                print("SpeciesNet Info/Warnings:\n", stderr)
            return output_json_path
        else:
            print("-" * 70)
            print("STEP 2: SPECIESNET FAILED.")
            # Existing error details will follow
            print("-" * 70)
            return None
    except FileNotFoundError:
        print(f"Error: '{python_executable} -m speciesnet.scripts.run_model' command not found.")
        print("Make sure SpeciesNet is installed and you are in the correct virtual environment,")
        print("and that the Python executable is in your PATH or correctly specified.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running SpeciesNet: {e}")
        return None

# %% CELL 4: Process Results Function
def process_speciesnet_results(predictions_json_path, target_species_keywords=None):
    """
    Loads SpeciesNet JSON output and prints detections, optionally filtering for target species.

    Args:
        predictions_json_path (str): Path to the SpeciesNet JSON output file.
        target_species_keywords (list, optional): List of keywords to filter species (e.g., ["badger", "meles"]).
                                                 Case-insensitive.
    """
    if not predictions_json_path or not os.path.exists(predictions_json_path):
        print(f"Error: Predictions JSON file not found at {predictions_json_path}")
        return

    print("\n" + "="*70)
    print("STEP 3: PROCESSING SPECIESNET RESULTS")
    print("="*70)
    print(f"  Reading predictions from: {predictions_json_path}")

    try:
        with open(predictions_json_path, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {predictions_json_path}")
        return
    
    print("\n--- SpeciesNet Results ---")
    if not results.get("predictions"):
        print("No predictions found in the output file.")
        return

    badger_detections_count = 0
    for item in results["predictions"]:
        filepath = item.get("filepath", "N/A")
        prediction = item.get("prediction", "N/A") # Final ensemble prediction
        prediction_score = item.get("prediction_score", 0.0)
        
        # Check top-5 classifications as well for more detail
        classifications = item.get("classifications", {}).get("classes", [])
        scores = item.get("classifications", {}).get("scores", [])

        detected = False
        if target_species_keywords:
            # Check ensemble prediction
            if any(keyword.lower() in prediction.lower() for keyword in target_species_keywords):
                detected = True
            else: # Check top-5 raw classifications
                for i, cls_name in enumerate(classifications):
                    if any(keyword.lower() in cls_name.lower() for keyword in target_species_keywords):
                        print(f"  Target keyword found in raw classification: {cls_name} (Score: {scores[i]:.2f}) for {filepath}")
                        detected = True # Count it even if not the top ensemble prediction
                        break
        else: # If no keywords, consider any animal detection as relevant for general overview
            if prediction != "blank" and prediction != "unknown" and prediction_score > 0.1: # Basic filter
                 detected = True

        if detected:
            badger_detections_count +=1
            print(f"File: {filepath}")
            print(f"  Ensemble Prediction: {prediction} (Score: {prediction_score:.2f})")
            if classifications:
                print("  Top Raw Classifications:")
                for i, cls_name in enumerate(classifications):
                    print(f"    - {cls_name}: {scores[i]:.4f}")
            print("-" * 20)

    if target_species_keywords:
        print(f"\nFound {badger_detections_count} frames with potential '{', '.join(target_species_keywords)}' detections.")
    else:
        print(f"\nProcessed {len(results['predictions'])} frames. Review output for detections.")

    print("-" * 70)
    print("STEP 3: RESULTS PROCESSING COMPLETE.")
    print("-" * 70)

# %% CELL 6: Visualize Detections (Optional)

def convert_json_paths_to_relative(input_json_path, base_path_for_images):
    """
    Reads a predictions JSON, converts absolute 'filepath' entries to be relative 
    to a given base_path, and saves to a temporary file.

    Args:
        input_json_path (str): Path to the original predictions JSON file.
        base_path_for_images (str): The base directory that filepaths should be relative to.

    Returns:
        str: Path to the temporary JSON file with relative paths, or None on error.
    """
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing JSON {input_json_path}: {e}")
        return None

    if 'predictions' not in data:
        print(f"'predictions' key not found in {input_json_path}")
        return None

    # Ensure base_path ends with a separator for reliable os.path.relpath
    normalized_base_path = os.path.join(os.path.abspath(base_path_for_images), "") 

    for pred in data['predictions']:
        if 'filepath' in pred:
            abs_filepath = os.path.abspath(pred['filepath'])
            try:
                # Make path relative
                relative_path = os.path.relpath(abs_filepath, normalized_base_path)
                pred['filepath'] = relative_path.replace('\\', '/') # Use forward slashes for consistency
            except ValueError as e:
                # This can happen if paths are on different drives on Windows
                print(f"Could not make path {abs_filepath} relative to {normalized_base_path}: {e}. Keeping absolute.")
                pred['filepath'] = abs_filepath.replace('\\', '/')
    
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8')
        json.dump(data, temp_file, indent=4)
        temp_file_path = temp_file.name
        temp_file.close()
        print(f"Converted JSON with relative paths saved to temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"Error writing temporary modified JSON: {e}")
        if temp_file:
            temp_file.close()
            os.unlink(temp_file.name) # Clean up temp file if created
        return None

def visualize_predictions(predictions_json_path, original_frames_folder, video_specific_base_output_dir):
    """
    Visualizes SpeciesNet predictions on the original frames using megadetector-utils.

    Args:
        predictions_json_path (str): Path to the SpeciesNet JSON output file.
        original_frames_folder (str): Path to the folder containing the original extracted frames.
        video_specific_base_output_dir (str): The base output directory for the current video.

    Returns:
        str: Path to the directory of visualized frames if successful, None otherwise.
    """
    print("\n" + "="*70)
    print("STEP 4: VISUALIZING DETECTIONS ON FRAMES")
    print("="*70)

    # Initialize temp_predictions_json_path here
    temp_predictions_json_path = None

    # This print statement was causing the error, ensure temp_predictions_json_path is defined
    # It will be properly set after convert_json_paths_to_relative is called.
    # For this initial print, we can show the original path and mention conversion.
    print(f"  Will use predictions from: {predictions_json_path} (after converting paths to relative)")
    print(f"  Reading original frames from: {original_frames_folder}")
    print(f"  Saving visualized frames to: {os.path.join(video_specific_base_output_dir, 'visualized_frames')}")

    if not os.path.exists(predictions_json_path):
        print(f"Predictions JSON file not found at {predictions_json_path}. Skipping visualization.")
        return None

    if not os.path.isdir(original_frames_folder):
        print(f"Original frames folder not found at {original_frames_folder}. Skipping visualization.")
        return None

    temp_predictions_json_path = convert_json_paths_to_relative(predictions_json_path, original_frames_folder)

    if not temp_predictions_json_path:
        print("Failed to convert JSON paths to relative. Skipping visualization.")
        return None

    visualization_output_dir = os.path.join(video_specific_base_output_dir, "visualized_frames")
    os.makedirs(visualization_output_dir, exist_ok=True)

    python_executable = "python" 

    cmd = [
        python_executable, "-m", "megadetector.visualization.visualize_detector_output",
        os.path.abspath(temp_predictions_json_path), 
        os.path.abspath(visualization_output_dir),
        "--images_dir", os.path.abspath(original_frames_folder),
        "--confidence", "0.1" 
    ]

    print(f"Running visualization command: {' '.join(cmd)}")
    success = False
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=".") 
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("-" * 70)
            print("STEP 4: VISUALIZATION COMPLETED SUCCESSFULLY.")
            print(f"  Visualized frames saved to: {visualization_output_dir}")
            print("-" * 70)
            success = True
            if stdout:
                 print("Visualization Stdout:\n", stdout)
            if stderr:
                print("Visualization Stderr/Info:\n", stderr)
        else:
            print("-" * 70)
            print("STEP 4: VISUALIZATION FAILED.")
            # Existing error details will follow
            print("-" * 70)
    except FileNotFoundError:
        print(f"Error: '{python_executable} -m megadetector.visualization.visualize_detector_output' command not found.")
        print("Make sure megadetector-utils is installed in your virtual environment.")
    except Exception as e:
        print(f"An unexpected error occurred during visualization: {e}")
    finally:
        if temp_predictions_json_path and os.path.exists(temp_predictions_json_path):
            try:
                os.unlink(temp_predictions_json_path)
                print(f"Cleaned up temporary JSON file: {temp_predictions_json_path}")
            except Exception as e_unlink:
                print(f"Error cleaning up temporary JSON file {temp_predictions_json_path}: {e_unlink}")
    
    return os.path.abspath(visualization_output_dir) if success else None

# %% CELL 7: Create Video from Frames
def create_video_from_frames(original_extracted_frames_dir, visualized_frames_dir, output_video_path, fps):
    """
    Creates a video from a folder of image frames, prioritizing visualized frames.
    If a visualized frame is not available for a corresponding original frame,
    the original frame is used instead, ensuring the output video has the same
    number of frames as were extracted.

    Args:
        original_extracted_frames_dir (str): Path to the folder containing ALL original extracted frames.
        visualized_frames_dir (str): Path to the folder containing frames with visualizations.
                                     Can be None if visualization failed or was skipped.
        output_video_path (str): Path to save the output video file (e.g., 'output.mp4').
        fps (float): Frames per second for the output video.
    """
    print("\n" + "="*70)
    print("STEP 5: CREATING OUTPUT VIDEO WITH DETECTIONS")
    print("="*70)
    print(f"  Reading original frames from: {original_extracted_frames_dir}")
    if visualized_frames_dir and os.path.isdir(visualized_frames_dir):
        print(f"  Using annotated frames from: {visualized_frames_dir}")
    else:
        print(f"  No valid annotated frames directory, using original frames only.")
    print(f"  Output video will be saved to: {output_video_path} at {fps} FPS.")

    if not os.path.isdir(original_extracted_frames_dir):
        print(f"Error: Original extracted frames folder not found at {original_extracted_frames_dir}")
        return False
    
    if fps <= 0:
        print(f"Error: Invalid FPS: {fps}. FPS must be positive.")
        return False

    original_image_filenames = sorted([
        f 
        for f in os.listdir(original_extracted_frames_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not original_image_filenames:
        print(f"Error: No image files found in {original_extracted_frames_dir}")
        return False

    try:
        first_frame_path = os.path.join(original_extracted_frames_dir, original_image_filenames[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            print(f"Error: Could not read the first frame: {first_frame_path}")
            return False
        height, width, layers = first_frame.shape
    except Exception as e:
        print(f"Error reading first frame ({first_frame_path}) to get dimensions: {e}")
        return False

    # fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        # print(f"Error: Could not open video writer for path {output_video_path} with codec XVID.")
        print(f"Error: Could not open video writer for path {output_video_path} with codec mp4v.")
        return False

    print(f"Processing {len(original_image_filenames)} frames for video creation (target dimensions: {width}x{height})...")
    frames_written = 0
    for i, image_filename in enumerate(original_image_filenames):
        frame_to_read_path = None

        if visualized_frames_dir:
            annotated_image_filename = "anno_" + image_filename
            potential_viz_frame_path = os.path.join(visualized_frames_dir, annotated_image_filename)
            
            base_name_orig, ext_orig = os.path.splitext(image_filename)
            annotated_base_name_jpg = "anno_" + base_name_orig + ".jpg"
            potential_viz_frame_path_jpg = os.path.join(visualized_frames_dir, annotated_base_name_jpg)
            
            if os.path.exists(potential_viz_frame_path):
                frame_to_read_path = potential_viz_frame_path
            elif os.path.exists(potential_viz_frame_path_jpg):
                 frame_to_read_path = potential_viz_frame_path_jpg

        if not frame_to_read_path:
            frame_to_read_path = os.path.join(original_extracted_frames_dir, image_filename)
        
        try:
            frame = cv2.imread(frame_to_read_path)
            if frame is not None:
                # Ensure frame dimensions match the video writer's expected dimensions
                current_height, current_width, _ = frame.shape
                if current_width != width or current_height != height:
                    print(f"Warning: Frame {image_filename} dimensions ({current_width}x{current_height}) differ from target ({width}x{height}). Resizing.")
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
                frames_written += 1
            else:
                print(f"Warning: Could not read frame {frame_to_read_path}, skipping.")
        except Exception as e:
            print(f"Warning: Error processing frame {frame_to_read_path}: {e}, skipping.")
        
        if (i+1) % 50 == 0: 
            print(f"  Processed {i+1}/{len(original_image_filenames)} frames... ({frames_written} written to video)")

    out.release()
    print("-" * 70)
    print("STEP 5: VIDEO CREATION COMPLETED SUCCESSFULLY.")
    print(f"  Output video: {output_video_path}")
    print("-" * 70)
    if frames_written < len(original_image_filenames):
        print(f"Warning: {len(original_image_filenames) - frames_written} frames were not written to the video.")
    if frames_written == 0 and len(original_image_filenames) > 0:
        print("Critical Warning: No frames were successfully written to the output video despite source frames being present.")
        return False
    return True

# %% CELL 5: Main Execution Block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos with SpeciesNet to detect animals.")
    parser.add_argument("video_path", nargs='?', default=DEFAULT_VIDEO_PATH,
                        help="Path to the video file to process.")
    parser.add_argument("--output_dir", default=BASE_OUTPUT_DIR,
                        help="Base directory to save extracted frames and results.")
    parser.add_argument("--interval", type=int, default=FRAME_INTERVAL_SECONDS,
                        help="Interval in seconds between extracted frames.")
    parser.add_argument("--country", default=COUNTRY_CODE,
                        help="Optional: ISO 3166-1 alpha-3 country code for geofencing (e.g., GBR).")

    args = parser.parse_args()

    if args.video_path == DEFAULT_VIDEO_PATH and not os.path.exists(DEFAULT_VIDEO_PATH):
        print(f"ERROR: Please provide a valid video_path argument or change DEFAULT_VIDEO_PATH in the script.")
        print(f"Example: python {__file__} /path/to/your/video.mp4")
        exit(1)

    video_filename_no_ext = os.path.splitext(os.path.basename(args.video_path))[0]
    
    video_specific_output_dir = os.path.join(args.output_dir, video_filename_no_ext)
    frames_output_folder = os.path.join(video_specific_output_dir, "extracted_frames")
    speciesnet_results_json = os.path.join(video_specific_output_dir, "speciesnet_predictions.json")
    # Construct a name for the final output video with detections
    output_video_file = os.path.join(video_specific_output_dir, f"{video_filename_no_ext}_detections.mp4")
    
    os.makedirs(video_specific_output_dir, exist_ok=True)
    os.makedirs(frames_output_folder, exist_ok=True)

    print("="*70)
    print("STARTING VIDEO PROCESSING PIPELINE")
    print("="*70)
    print(f"Input Video: {args.video_path}")
    print(f"Base Output Directory: {args.output_dir}")
    print(f"Frames will be extracted to: {frames_output_folder}")
    print(f"SpeciesNet JSON results will be saved to: {speciesnet_results_json}")
    print(f"Final video with detections will be saved to: {output_video_file}")
    print(f"Frame extraction interval: {'All frames' if args.interval == 0 else str(args.interval) + ' second(s)'}")

    print("\n" + "="*70)
    print("STEP 1: EXTRACTING FRAMES FROM VIDEO")
    print("="*70)
    # 1. Extract Frames
    extracted_frame_paths, original_fps = extract_frames(args.video_path, frames_output_folder, interval_seconds=args.interval)

    if not extracted_frame_paths or original_fps is None:
        print("No frames were extracted or FPS could not be determined. Exiting.")
        exit(1)

    if extracted_frame_paths and original_fps is not None:
        print("-" * 70)
        print(f"STEP 1: FRAME EXTRACTION COMPLETE. ({len(extracted_frame_paths)} frames extracted at {original_fps} FPS)")
        print("-" * 70)

    # 2. Run SpeciesNet
    predictions_file = run_speciesnet_on_frames(
        frames_folder_path=frames_output_folder,
        output_json_path=speciesnet_results_json,
        country_code=args.country
    )

    # 3. Process Results and Visualize
    if predictions_file:
        target_keywords = ["badger", "meles"]
        process_speciesnet_results(predictions_file, target_species_keywords=target_keywords)
        
        visualized_frames_dir = visualize_predictions(
            predictions_json_path=predictions_file, 
            original_frames_folder=frames_output_folder, 
            video_specific_base_output_dir=video_specific_output_dir
        )

        if visualized_frames_dir and original_fps > 0:
            # 4. Create video from visualized frames
            create_video_from_frames(
                original_extracted_frames_dir=frames_output_folder, 
                visualized_frames_dir=visualized_frames_dir, 
                output_video_path=output_video_file, 
                fps=original_fps
            )
        elif not visualized_frames_dir:
            print("Visualization of frames failed or was skipped. Cannot create output video.")
        elif not original_fps > 0:
            print(f"Original FPS is not valid ({original_fps}). Cannot create output video.")

    else:
        print("SpeciesNet did not produce an output file. Skipping results processing, visualization, and video creation.")

    print("\n" + "="*70)
    print(f"ALL PROCESSING STAGES COMPLETE FOR: {args.video_path}")
    print("="*70)
