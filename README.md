
# Codsoft AI Internship Projects

This repository contains projects completed during my AI internship at Codsoft. The projects focus on artificial intelligence applications.

## Projects

#### Face Recognition and Detection using OpenCV and Haar Cascade Algorithm

This project implements a face recognition system using the OpenCV library and Haar Cascade classifier. It generates live datasets using a web camera.

## Steps and Processes
1. Install Dependencies
Ensure Python is installed, then install OpenCV:
```bash
pip install opencv-python
```

## 2. Haar Cascade Classifier
Download the pre-trained Haar Cascade classifier for face detection from the OpenCV GitHub repository.

## 3. Capture Live Dataset
Create a script to capture images from the webcam, labeling and storing them for training:
```python
import cv2

def generate_dataset(img, id, img_id):
    cv2.imwrite(f"data/user.{id}.{img_id}.jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf=None):
    if img is None or img.size == 0:
        print("Error: The image is empty or not loaded correctly.")
        return []

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if clf is not None:
            id, _ = clf.predict(gray_img[y:y+h, x:x+w])
            if id == 1:
                cv2.putText(img, "Ayuska", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img

def detect(img, faceCascade, eyeCascade, img_id):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)
    return img

 Load Haar cascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the trained recognizer model
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

img_id = 0

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Error: Unable to capture video")
        break

    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Detection", img)
    img_id += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()

```

## 4. Preprocess and Label Data
Organize the captured images into directories.

## 5. Train the Model
Use OpenCV's `face.LBPHFaceRecognizer_create()` method to train a face recognition model with the labeled data.

## 6. Recognize Faces
Implement face recognition using the trained model:
```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # Display the name
        cv2.putText(frame, str(id), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 7. Evaluate and Optimize
Test the system with live input, refining the model as needed for accuracy and performance.

## Conclusion
This README provides an overview of implementing a face recognition system using OpenCV and Haar Cascade, guiding through dependency installation, data collection, training, and real-time face recognition.


# 2. Tic Tac Toe (Human vs Computer)

This README provides a comprehensive guide to setting up and running a Tic Tac Toe game where a human player competes against an AI opponent. 
The AI uses the minimax algorithm to make optimal moves.
## Steps and Processes

## 1. Setup and Dependencies

Ensure you have Python installed.

### 2. Create the Board

```python
def create_board():
    return [[" " for _ in range(3)] for _ in range(3)]

def print_board(board):
    for row in board:
        print("|".join(row))
        print("-----")
```

### 3. Validate Moves

```python
def is_valid_move(board, move):
    row, col = move
    return 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == " "

def make_move(board, player, move):
    row, col = move
    board[row][col] = player
```

### 4. Switch Player

```python
def switch_player(player):
    return "O" if player == "X" else "X"

def is_game_over(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != " ":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != " ":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]
    if all(cell != " " for row in board for cell in row):
        return "Draw"
    return None
```

### 5. Minimax Algorithm

```python
def get_ai_move(board, ai_player):
    def minimax(board, player, alpha, beta):
        game_over_result = is_game_over(board)
        if game_over_result:
            if game_over_result == ai_player:
                return (1, None)
            elif game_over_result == switch_player(ai_player):
                return (-1, None)
            else:
                return (0, None)
        best_move = None
        if player == ai_player:
            best_score = -float('inf')
            for row in range(3):
                for col in range(3):
                    if is_valid_move(board, (row, col)):
                        board[row][col] = player
                        score, _ = minimax(board, switch_player(player), alpha, beta)
                        board[row][col] = " "
                        if score > best_score:
                            best_score = score
                            best_move = (row, col)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
        else:
            best_score = float('inf')
            for row in range(3):
                for col in range(3):
                    if is_valid_move(board, (row, col)):
                        board[row][col] = player
                        score, _ = minimax(board, switch_player(player), alpha, beta)
                        board[row][col] = " "
                        if score < best_score:
                            best_score = score
                            best_move = (row, col)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
        return best_score, best_move
    _, best_move = minimax(board, ai_player, -float('inf'), float('inf'))
    return best_move
```

### 6. Get Human Move

```python
def get_human_move(board):
    while True:
        try:
            row = int(input("Enter row (1-3): ")) - 1
            col = int(input("Enter column (1-3): ")) - 1
            if is_valid_move(board, (row, col)):
                return (row, col)
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Please enter numbers.")
```

### 7. Main Game Loop

```python
def main():
    board = create_board()
    current_player = "X"

    while True:
        print_board(board)

        if current_player == "O":
            move = get_ai_move(board, current_player)
            print("AI played:", move)
        else:
            move = get_human_move(board)

        make_move(board, current_player, move)

        game_over_result = is_game_over(board)
        if game_over_result:
            print_board(board)
            if game_over_result == "Draw":
                print("It's a draw!")
            else:
                print(f"{game_over_result} wins!")
            break

        current_player = switch_player(current_player)

if __name__ == "__main__":
    main()
```

# Rule-Based Chatbot

This project implements a rule-based chatbot that interacts with users based on predefined rules and patterns.

## Features

- **Pattern Matching**: Uses simple pattern matching to identify user intent and provide appropriate responses.
- **Predefined Responses**: Responds to user inputs based on a set of predefined rules.
- **Extendable**: Easily extendable to include more patterns and responses.

## Getting Started

### Prerequisites

- Python 3.x

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/rule-based-chatbot.git
    cd rule-based-chatbot
    ```

2. Install dependencies (if any):
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the chatbot:
    ```bash
    python chatbot.py
    ```

2. Start interacting with the chatbot by typing messages.

## Acknowledgments
- Codsoft for providing the internship opportunity
