from datetime import datetime, timedelta
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

# Emotion-time record
class EmotionWithTime:
    def __init__(self, emotion: str, time: timedelta):
        self.emotion = emotion
        self.time = time

# Accomulator-class for emotions during the session
class EmotionHistory:
    def __init__(self):
        self.history: dict[datetime, dict[str, float]] = {}         # all emations in timeline order
        self.dominant_emotions: dict[datetime, str] = {}            # all dominant emotions in timeline order

    def add(self, date: datetime, emotions: dict[str, float], dominant_emotion: str):
        self.history[date] = emotions
        self.dominant_emotions[date] = dominant_emotion

    def get_emotions_time(self):                                    # list of all emotions and summary time during the session
        emotions_time: dict[str, timedelta] = {}
        items = sorted(self.dominant_emotions.items())
        
        prev_date, prev_emotion = items[0]
        for dt, emotion in items[1:]:
            emotions_time[prev_emotion] = (
                emotions_time.get(prev_emotion, timedelta()) + (dt - prev_date)
            ) 
            prev_date, prev_emotion = dt, emotion

        emotions_class_time = {               # list of negative, positive and neutral emotions
            'negative': (                                           # with calculated summary time during the session 
                emotions_time.get('disgust', timedelta()) +
                emotions_time.get('angry', timedelta()) +
                emotions_time.get('fear', timedelta()) +
                emotions_time.get('sad', timedelta())
            ),
            'positive': emotions_time.get('happy', timedelta()),
            'neutral':  (
                emotions_time.get('surprise', timedelta()) +
                emotions_time.get('neutral', timedelta())
            )
        }

        return emotions_time, emotions_class_time

    def session_summary(self):                                       # collect summary and show result with draft
        emotions_time, emotion_class_time = self.get_emotions_time()
        
        first_emotion, first_time = next(iter(emotions_time.items()))
        min_emotion = EmotionWithTime(first_emotion, first_time)
        max_emotion = EmotionWithTime(first_emotion, first_time)
        
        session_time = first_time
        
        first_element = True
        for emotion, time in emotions_time.items():
            if first_element:
                first_element = False
                continue

            if min_emotion.time >= time:
                min_emotion = EmotionWithTime(emotion, time)
            
            if max_emotion.time <= time:
                max_emotion = EmotionWithTime(emotion, time)
            
            session_time += time

        # make result text description
        result = (f'''
During the session, the client most often experienced {get_emotion_info(max_emotion.emotion)} and most rare {get_emotion_info(min_emotion.emotion)}.

List of emotions and their total duration per session:\n''')
        
        for emotion, time in emotion_class_time.items():                            # display a breakdown of time spent on each emotion category
            result += f'{emotion} : {format_timedelta(time)}\n'

        result += '\n\n' + 'A breakdown of emotions and their total duration per session:\n'
        for emotion, time in emotions_time.items():                                 # display a breakdown of time spent on each emotion
            result += f'{emotion} : {format_timedelta(time)}\n'

        return result + '\n' + f'Summary session time: {format_timedelta(session_time)}'  # calculating the total session time and displaying the result

    def do_chart(self):                         # Plotting a graph showing changes in emotions over the course of a session and displaying the result
        items = sorted(self.history.items())

        times = [t for t, _ in items]
        emotions = list(items[0][1].keys())

        for emotion in emotions:
            values = [emotions_dict[emotion] for _, emotions_dict in items]
            plt.plot(times, values, label=emotion)

        plt.legend()                                # start of graph display
        plt.xlabel("Время")                         # X-axis to represent time
        plt.ylabel("Вероятность")                   # Y-axis to represent time
        plt.title("График эмоций за сессию")        # add title
        plt.xticks(rotation=45)                     # rotate the labels on the X-axis marks by 45 degrees
        plt.tight_layout()                          # set the graph rendering type
        plt.subplots_adjust(bottom=0.4)             # set the indentation from the grid for the next element

        plt.figtext(0.01, 0.01, self.session_summary(), ha="left", fontsize=8)  # Display the text of the emotion analysis
        plt.show()                                  # show result window

# function for outputting the time in text format
def format_datetime(dt: datetime) -> str:
    return dt.strftime(time_format)[:-3]

# output of the calculated time in hours, minutes, seconds and milliseconds
def format_timedelta(td: timedelta) -> str:
    total_ms = int(td.total_seconds() * 1000)

    hours = total_ms // (3600 * 1000)
    minutes = (total_ms % (3600 * 1000)) // (60 * 1000)
    seconds = (total_ms % (60 * 1000)) // 1000
    ms = total_ms % 1000

    if hours > 0:
        return f"{hours} hours {minutes} minutes {seconds} seconds"
    elif minutes > 0:
        return f"{minutes} minutes {seconds} seconds"
    elif seconds > 0:
        return f"{seconds} seconds"
    else:
        return f"{ms} milliseconds"

# face frame coordinate detection function    
def get_coords_of_face(frame):
    face_location = DeepFace.extract_faces(frame, enforce_detection=False)
    face = face_location[0].get('facial_area')
    return face.get('x', 0), face.get('y', 0), face.get('w', 0), face.get('h', 0)

def get_emotion_info(emotion):
    if emotion in positive_emotions:
        result = '"positive"'
    elif emotion in negative_emotions:
        result = '"negative"'
    elif emotion in neutral_emotions:
        result = '"neutral"'
    else:
        result = '"unknown"'
    
    return result + f' emotion "{emotion}"'

############################################################

neutral_emotions = ('surprise', 'neutral')
positive_emotions = ('happy')
negative_emotions = ('disgust', 'angry', 'fear', 'sad')

time_format = "%Y-%m-%d %H:%M:%S.%f"
emotion_history = EmotionHistory()

print("Streaming started")
video_capture = cv2.VideoCapture(0)

# start video-stream session (press ESC for end)
while True:
    ret, frame = video_capture.read()

    result = DeepFace.analyze(
        frame,
        actions=['emotion'],
        detector_backend='opencv',
        enforce_detection=False
    )

    dominant_emotion = result[0]['dominant_emotion']
    emotion_history.add(datetime.now(), result[0]['emotion'], dominant_emotion)

    x, y, w, h = get_coords_of_face(frame)
        
    cv2.putText(frame, dominant_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord(chr(27)):
        break

video_capture.release()
cv2.destroyAllWindows()
emotion_history.do_chart()