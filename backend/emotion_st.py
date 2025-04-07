import json
import os
from datetime import datetime
from collections import defaultdict, Counter
from datetime import datetime
import json

DATA_FILE = "data/emotions.json"

def save_emotion(user_id, emotion, confidence):
    entry = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "emotion": emotion,
        "confidence": confidence
    }

    # Make sure the data folder and file exist
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump([], f)

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    data.append(entry)

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "success", "data": entry}

TIME_BLOCKS = [
    ("8am-11am", 8, 11),
    ("11am-2pm", 11, 14),
    ("2pm-5pm", 14, 17),
    ("5pm-8pm", 17, 20),
    ("8pm-11pm", 20, 23)
]
def get_emotion_heatmap(user_id):
    with open("data/emotions.json", "r") as f:
        data = json.load(f)

    # Filter for current user
    user_entries = [entry for entry in data if entry["user_id"] == user_id]

    # Prepare output structure
    grouped = defaultdict(lambda: defaultdict(list))  # grouped[day][block] = list of emotions

    for entry in user_entries:
        ts = datetime.fromisoformat(entry["timestamp"])
        day = ts.strftime("%A")  # "Monday", "Tuesday", etc.
        hour = ts.hour
        emotion = entry["emotion"]

        # Match the time block
        for label, start, end in TIME_BLOCKS:
            if start <= hour < end:
                grouped[day][label].append(emotion)
                break

    # Get dominant emotion for each block
    final = {}
    for day, blocks in grouped.items():
        final[day] = {}
        for block, emotions in blocks.items():
            most_common = Counter(emotions).most_common(1)[0][0]
            final[day][block] = most_common

    return final