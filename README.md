# Speech Emotion Recognition API

This API provides endpoints for processing text and audio files to detect emotions using generative AI models.

## Endpoints

### 1. `/convo/process-text`

**Method:** POST

**Description:** Processes a text input and returns a generative AI response.

**Request Body:**
```json
{
  "text": "Your input text here"
}
```

**Response:**
```json
{
  "response": "Generative AI response text"
}
```

### 2. `/convo/process-audio`

**Method:** POST

**Description:** Uploads an audio file, processes it with generative AI, and returns the response.

**Form Data:**
- `file`: The audio file to be uploaded.
- `responses`: A string of previous responses separated by `|`.

**Response:**
```json
{
  "message": "File uploaded and processed",
  "file_path": "unique_filename.extension",
  "response": "Generative AI response text"
}
```

### 3. `/convo/get-audio/{filename}`

**Method:** GET

**Description:** Retrieves the audio file by filename.

**Path Parameter:**
- `filename`: The name of the audio file to retrieve.

**Response:**
- Returns the audio file if found.
- If not found:
```json
{
  "error": "File not found"
}
```

### 4. `/auth/signup`

**Method:** POST

**Description:** Registers a new user.

**Request Body:**
```json
{
  "username": "user123",
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "message": "User registered successfully"
}
```

### 5. `/auth/login`

**Method:** POST

**Description:** Logs in a user and returns an access token.

**Request Body:**
```json
{
  "username": "user123",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "jwt_token",
  "token_type": "bearer"
}
```

### 6. `/protected`

**Method:** GET

**Description:** A protected route that requires a valid token.

**Response:**
```json
{
  "message": "Welcome, {username}! This is a protected route."
}
```

### 7. `/predict-indian`

**Method:** POST

**Description:** Uploads an audio file, processes it to predict emotion for Indian accent, and returns the emotion.

**Form Data:**
- `file`: The audio file to be uploaded.

**Response:**
```json
{
  "emotion": "predicted_emotion"
}
```

### 8. `/predict-emotion/`

**Method:** POST

**Description:** Uploads a WAV audio file, processes it to predict emotion, and returns the emotion.

**Form Data:**
- `file`: The WAV audio file to be uploaded.

**Response:**
```json
{
  "emotion": "predicted_emotion"
}
```

## Setup

1. Clone the repository.
2. Install the required dependencies.
3. Configure the environment variables.
4. Run the application.

## Dependencies

- FastAPI
- Motor
- Pydantic
- Google Generative AI
- Pydub
- Librosa
- TensorFlow
- Uvicorn

## Running the Application

```bash
uvicorn main:app --reload
```

## License

This project is licensed under the MIT License.