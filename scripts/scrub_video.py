import cv2
import os

def strip_metadata(input_path, output_path):
    print(f"[*] Reading raw video: {input_path}")
    
    # Open the video with OpenCV (which naturally ignores Apple metadata)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[!] Error: Could not open video.")
        return
        
    # Get basic video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up the writer for a clean MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("[*] Scrubbing metadata and rewriting frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"[*] SUCCESS: Clean video saved to: {output_path}")

if __name__ == "__main__":
    raw_video = "data/raw_video/grasp_test.mp4"
    clean_video = "data/raw_video/grasp_test_clean.mp4"
    
    if os.path.exists(raw_video):
        strip_metadata(raw_video, clean_video)
    else:
        print(f"[!] Input video not found at: {os.path.abspath(raw_video)}")