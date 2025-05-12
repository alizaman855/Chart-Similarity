import os
import cv2
import numpy as np
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import json
import threading

# Lock for thread safety when using matplotlib
matplotlib_lock = threading.Lock()

def find_most_similar_charts_in_video(video_path, output_dir, fps=1.0, progress_callback=None):
    """
    Analyzes video to find frames where charts on left and right sides show similar patterns.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save results
        fps: Frames per second to process (lower = more accurate but slower)
        progress_callback: Function to call with progress updates (0-100)
        
    Returns:
        Dictionary with analysis results
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps_video
    
    # Calculate frame interval based on desired fps
    frame_interval = int(fps_video / fps)
    if frame_interval < 1:
        frame_interval = 1
    
    # Prepare results
    frames_data = []
    similarity_scores = []
    frame_times = []
    
    # Process video frames
    frame_num = 0
    processed_count = 0
    top_frames = []
    
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every nth frame according to the frame_interval
        if frame_num % frame_interval == 0:
            # Calculate progress
            progress = min(100, int((frame_num / total_frames) * 100))
            if progress_callback:
                progress_callback(progress)
            
            # Get frame time in seconds
            frame_time = frame_num / fps_video
            
            # Process the frame to find similarity between left and right charts
            try:
                result = process_frame(frame, frame_num, frame_time, output_dir)
                frames_data.append(result)
                similarity_scores.append(result['similarity'])
                frame_times.append(frame_time)
                
                # Keep track of top frames by similarity
                if len(top_frames) < 10:
                    top_frames.append(result)
                    # Sort by similarity (highest first)
                    top_frames.sort(key=lambda x: x['similarity'], reverse=True)
                elif result['similarity'] > top_frames[-1]['similarity']:
                    # Replace the lowest similarity frame
                    top_frames[-1] = result
                    # Re-sort
                    top_frames.sort(key=lambda x: x['similarity'], reverse=True)
            except Exception as e:
                print(f"Error processing frame {frame_num}: {str(e)}")
            
            processed_count += 1
        
        frame_num += 1
    
    # Release the video
    cap.release()
    
    # Generate similarity plot
    similarity_plot_path = generate_similarity_plot(similarity_scores, frame_times, top_frames, output_dir)
    
    # Prepare results
    results = {
        'video_path': str(video_path),
        'output_dir': str(output_dir),
        'total_frames': total_frames,
        'processed_frames': processed_count,
        'fps': fps,
        'duration': duration,
        'top_frames': top_frames,
        'similarity_plot': similarity_plot_path
    }
    
    # Save results to file
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(prepare_results_for_json(results), f)
    
    # Set progress to 100% when finished
    if progress_callback:
        progress_callback(100)
    
    return results


def process_frame(frame, frame_number, frame_time, output_dir):
    """
    Process a single frame to find similarity between left and right charts.
    
    Args:
        frame: The video frame (OpenCV image)
        frame_number: Frame number in the video
        frame_time: Time of the frame in seconds
        output_dir: Directory to save results
        
    Returns:
        Dictionary with frame data and similarity score
    """
    # Create frame output directory
    frame_dir = Path(output_dir)
    
    # Format frame number with leading zeros
    frame_num_str = f"{frame_number:06d}"
    
    # Save the original frame
    frame_path = frame_dir / f"frame_{frame_num_str}.jpg"
    cv2.imwrite(str(frame_path), frame)
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Split frame in half (left and right sides)
    mid_point = width // 2
    left_frame = frame[:, :mid_point]
    right_frame = frame[:, mid_point:]
    
    # Save split frames
    left_path = frame_dir / f"left_{frame_num_str}.jpg"
    right_path = frame_dir / f"right_{frame_num_str}.jpg"
    cv2.imwrite(str(left_path), left_frame)
    cv2.imwrite(str(right_path), right_frame)
    
    # Extract chart areas (simplified - assuming charts are green lines)
    # In a real implementation, you would need more sophisticated chart detection
    left_chart = extract_chart_area(left_frame)
    right_chart = extract_chart_area(right_frame)
    
    # Calculate similarity between charts
    similarity = calculate_chart_similarity(left_chart, right_chart)
    
    # Generate overlay visualization
    overlay_path = generate_overlay_visualization(left_chart, right_chart, frame_dir, frame_num_str)
    
    # Generate profile comparison
    profile_path = generate_profile_comparison(left_chart, right_chart, frame_dir, frame_num_str)
    
    # Return frame data
    return {
        'frame_number': frame_number,
        'time': frame_time,
        'similarity': similarity,
        'frame_path': str(frame_path.relative_to(Path(output_dir).parent)),
        'left_path': str(left_path.relative_to(Path(output_dir).parent)),
        'right_path': str(right_path.relative_to(Path(output_dir).parent)),
        'overlay_path': str(overlay_path.relative_to(Path(output_dir).parent)),
        'profile_path': str(profile_path.relative_to(Path(output_dir).parent))
    }


def extract_chart_area(frame):
    """
    Extract the chart area from a frame, focusing on green chart lines.
    
    Args:
        frame: Input frame (OpenCV image)
        
    Returns:
        Processed chart image
    """
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define green color range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply mask to get only green parts
    chart = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary image
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    return binary


def calculate_chart_similarity(chart1, chart2):
    """
    Calculate similarity between two chart images.
    
    Args:
        chart1: First chart image
        chart2: Second chart image
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Resize if dimensions don't match
    if chart1.shape != chart2.shape:
        chart2 = cv2.resize(chart2, (chart1.shape[1], chart1.shape[0]))
    
    # Calculate SSIM
    ssim_score, _ = ssim(chart1, chart2, full=True)
    
    # Calculate correlation
    # Flatten arrays for correlation
    flat1 = chart1.flatten()
    flat2 = chart2.flatten()
    
    # Only calculate correlation if there are enough non-zero pixels
    if np.count_nonzero(flat1) > 10 and np.count_nonzero(flat2) > 10:
        try:
            corr, _ = pearsonr(flat1, flat2)
            # Handle NaN (can happen with constant values)
            if np.isnan(corr):
                corr = 0
        except Exception:
            corr = 0
    else:
        corr = 0
    
    # Combine scores (weighted average)
    similarity = (0.7 * ssim_score) + (0.3 * (corr + 1) / 2)
    
    return similarity


def generate_overlay_visualization(chart1, chart2, output_dir, frame_num_str):
    """
    Generate a visualization showing both charts overlaid.
    
    Args:
        chart1: First chart image
        chart2: Second chart image
        output_dir: Directory to save the visualization
        frame_num_str: Frame number string
        
    Returns:
        Path to the saved visualization
    """
    with matplotlib_lock:
        # Create a figure
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Resize if dimensions don't match
        if chart1.shape != chart2.shape:
            chart2 = cv2.resize(chart2, (chart1.shape[1], chart1.shape[0]))
        
        # Create RGB version of the charts
        height, width = chart1.shape
        rgb_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Red channel for chart1
        rgb_overlay[chart1 > 0, 0] = 255
        
        # Blue channel for chart2
        rgb_overlay[chart2 > 0, 2] = 255
        
        # Both charts overlap in purple (red + blue)
        
        # Display the overlay
        ax.imshow(rgb_overlay)
        ax.set_title(f'Chart Overlay - Frame {frame_num_str}')
        ax.axis('off')
        
        # Save the figure
        output_path = Path(output_dir) / f"overlay_{frame_num_str}.png"
        fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path


def generate_profile_comparison(chart1, chart2, output_dir, frame_num_str):
    """
    Generate a visualization comparing the profiles of both charts in horizontal orientation.
    
    Args:
        chart1: First chart image
        chart2: Second chart image
        output_dir: Directory to save the visualization
        frame_num_str: Frame number string
        
    Returns:
        Path to the saved visualization
    """
    with matplotlib_lock:
        # Create a figure
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Resize if dimensions don't match
        if chart1.shape != chart2.shape:
            chart2 = cv2.resize(chart2, (chart1.shape[1], chart1.shape[0]))
        
        # Calculate vertical profiles (sum along horizontal axis)
        profile1 = np.sum(chart1, axis=1) / 255
        profile2 = np.sum(chart2, axis=1) / 255
        
        # Normalize
        if np.max(profile1) > 0:
            profile1 = profile1 / np.max(profile1)
        if np.max(profile2) > 0:
            profile2 = profile2 / np.max(profile2)
        
        # Create y-axis values (row indices)
        y = np.arange(len(profile1))
        
        # Plot profiles horizontally (swapping x and y axes)
        ax.plot(y, profile1, 'r-', label='Left Chart')
        ax.plot(y, profile2, 'b-', label='Right Chart')
        
        # Set labels and title
        ax.set_ylabel('Normalized Intensity')
        ax.set_xlabel('Vertical Position')
        ax.set_title(f'Price Movement Comparison - Frame {frame_num_str}')
        ax.legend()
        
        # Adjust the layout
        fig.tight_layout()
        
        # Save the figure
        output_path = Path(output_dir) / f"profile_{frame_num_str}.png"
        fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path


def generate_similarity_plot(similarity_scores, frame_times, top_frames, output_dir):
    """
    Generate a plot showing similarity scores over time.
    
    Args:
        similarity_scores: List of similarity scores
        frame_times: List of frame times
        top_frames: List of top frames by similarity
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot
    """
    with matplotlib_lock:
        # Create a figure
        fig = Figure(figsize=(12, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Plot similarity scores
        ax.plot(frame_times, similarity_scores, 'b-', alpha=0.7)
        
        # Plot top frames as vertical lines
        for frame in top_frames[:5]:  # Top 5 frames
            ax.axvline(x=frame['time'], color='r', linestyle='--', alpha=0.7)
            ax.text(frame['time'], 0.1, f"{frame['time']:.1f}s", rotation=90, verticalalignment='bottom')
        
        # Set labels and title
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Chart Similarity Throughout Video')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Save the figure
        output_path = Path(output_dir) / "similarity_plot.png"
        fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path.relative_to(Path(output_dir).parent)


def prepare_results_for_json(results):
    """
    Prepare results dictionary for JSON serialization.
    Converts any non-serializable objects to strings.
    
    Args:
        results: Results dictionary
        
    Returns:
        JSON-serializable dictionary
    """
    # Create a copy to avoid modifying the original
    serializable = {}
    
    for key, value in results.items():
        # Handle non-serializable types
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.integer):
            serializable[key] = int(value)
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        elif isinstance(value, list):
            # Process lists recursively
            serializable[key] = []
            for item in value:
                if isinstance(item, dict):
                    # Process dictionaries in lists
                    serializable[key].append(prepare_results_for_json(item))
                else:
                    # Add other items as is
                    serializable[key].append(item)
        elif isinstance(value, dict):
            # Process dictionaries recursively
            serializable[key] = prepare_results_for_json(value)
        else:
            # Add other types as is
            serializable[key] = value
    
    return serializable