import cv2
import pandas as pd
import keyboard
import numpy as np

# Define video path and CSV file
video_path = 'video.mp4'  # Change to your video file path
csv_file = 'annotated_frames.csv'

# Initialize variables
frame_indexes = set()  # List to store the annotated frame indices

paused = False
current_frame_index = 0

# Open the video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Check if video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create or load CSV file for annotations
try:
    # Load existing frame annotations
    df = pd.read_csv(csv_file)
    frame_indexes = df['Frame_Index'].tolist()
except FileNotFoundError:
    # If no CSV exists, create a new one
    df = pd.DataFrame(columns=['Frame_Index'])
    frame_indexes = []
frame_indexes = set(frame_indexes)


def find_largest_smaller_than_a(lst, a):
    # Initialize a variable to track the largest number smaller than `a`
    largest = None

    # Iterate over the list to find the largest number smaller than `a`
    for num in lst:
        if num < a and (largest is None or num > largest):
            largest = num

    # Return the result
    return largest

def save_annotations():
    """ Save the annotated frames to a CSV file """
    df = pd.DataFrame({'Frame_Index': sorted(frame_indexes)})
    df.to_csv(csv_file, index=False)
    print(f"Annotations saved to {csv_file}")

def s_callback(e):
    if current_frame_index not in frame_indexes:
        frame_indexes.add(current_frame_index)
        print(f"Frame {current_frame_index} annotated.")
    else:
        frame_indexes.remove(current_frame_index)
    save_annotations()

def w_callback(e):
    global current_frame_index
    current_frame_index = max(0, current_frame_index - 1)

def f_callback(e):
    global current_frame_index
    current_frame_index = min(total_frames - 1, current_frame_index + 1)
def r_callback(e):
    global current_frame_index
    current_frame_index = min(total_frames - 1, current_frame_index - 10)


keyboard.on_press_key('s', s_callback)
keyboard.on_press_key('w', w_callback)
keyboard.on_press_key('f', f_callback)
keyboard.on_press_key('r', r_callback)

# Main loop to play, control, and annotate the video
fps = 25
dist = 100  # [m]
font_scale = 0.8
while True:
    # Set the current frame and read it
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    frame = scaled_image = cv2.resize(frame, (int(width * 1.5), int(height * 1.5)))
    height, width, _ = frame.shape

    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Display the frame

    # Handle user inputs for video controls
    if keyboard.is_pressed('q'):  # Quit the video
        save_annotations()
        break
    elif keyboard.is_pressed('p'):  # Pause the video
        paused = True

    # Display controls on the screen
    if paused:
        cv2.putText(frame, "PAUSED", (0, int(0.83*height) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        current_frame_index += 1
    if current_frame_index in frame_indexes:
        cv2.putText(frame, "SELECTED", (0, int(0.63*height) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {current_frame_index}/{total_frames}", (0, int(0.028*height) + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 2)
    cv2.putText(frame, "Press 's' to select frame, 'q' to quit", (0, int(0.083*height) + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 2)
    cv2.putText(frame, "Controls: 'p' Pause, 'r' Rewind, 'f' Forward, 'w' Move Back", (0, int(0.139*height) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    last_selected_frame = find_largest_smaller_than_a(list(frame_indexes), current_frame_index)
    if last_selected_frame is None:
        last_selected_frame = 0
    estimated_speed = 60*60*dist/((current_frame_index - last_selected_frame)/fps)/1000  # [km/h]
    cv2.putText(frame, f"estimated speed:{estimated_speed:.02f}[km/h]", (0, int(0.193*height) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    cv2.imshow('Video', frame)
    # print(current_frame_index)
    # Break the loop on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        save_annotations()
        break

    # If paused, wait for user input
    if paused:
        while paused:
            if keyboard.is_pressed('t'):  # Resume the video
                paused = False
                print("Playing")
            elif keyboard.is_pressed('m'):  # Jump to a specific frame (manual input)
                try:
                    frame_index_input = int(input(f"Enter frame index (0 to {total_frames - 1}): "))
                    if 0 <= frame_index_input < total_frames:
                        current_frame_index = frame_index_input
                    else:
                        print("Invalid frame index.")
                except ValueError:
                    print("Please enter a valid integer.")
                break
            elif keyboard.is_pressed('r'):  # Rewind (go back 10 frames)
                # current_frame_index = max(0, current_frame_index - 10)
                break
            elif keyboard.is_pressed('f'):  # Move forward one frame
                # current_frame_index = min(total_frames - 1, current_frame_index + 1)
                break
            elif keyboard.is_pressed('s'):  # Select the current frame
                break
            elif keyboard.is_pressed('w'):  # Move backward one frame
                # current_frame_index = max(0, current_frame_index - 1)
                break
            elif keyboard.is_pressed('p'):  # Move backward one frame
                # current_frame_index = max(0, current_frame_index - 1)
                break
            elif keyboard.is_pressed('q'):  # Escape to quit during pause
                save_annotations()
                break
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                save_annotations()
                break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
