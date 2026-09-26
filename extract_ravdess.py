import zipfile
import os
import shutil

def extract_emotions():
    zip_path = "ravdess.zip"
    if not os.path.exists(zip_path):
        print("Zip file not found yet.")
        return
        
    print("Extracting selected files from RAVDESS...")
    
    os.makedirs("demo_audio", exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        
        try:
            # 03 = happy, 04 = sad, 05 = angry
            happy_file = next(f for f in files if "-03-01-03-" in f)
            sad_file = next(f for f in files if "-03-01-04-" in f)
            angry_file = next(f for f in files if "-03-01-05-" in f)
            
            zip_ref.extract(happy_file, "demo_audio_temp")
            zip_ref.extract(sad_file, "demo_audio_temp")
            zip_ref.extract(angry_file, "demo_audio_temp")
            
            shutil.move(os.path.join("demo_audio_temp", happy_file), "demo_audio/demo_emotion_happy.wav")
            shutil.move(os.path.join("demo_audio_temp", sad_file), "demo_audio/demo_emotion_sad.wav")
            shutil.move(os.path.join("demo_audio_temp", angry_file), "demo_audio/demo_emotion_angry.wav")
            
            shutil.rmtree("demo_audio_temp")
            print("Successfully extracted real RAVDESS emotion samples!")
        except StopIteration:
            print("Could not find the expected RAVDESS files inside the zip.")

if __name__ == "__main__":
    extract_emotions()
