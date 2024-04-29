import tempfile
import subprocess
import os, stat
import uuid
from googletrans import Translator
# from TTS.api import TTS
import ffmpeg
from faster_whisper import WhisperModel
from zipfile import ZipFile
import shlex
from tqdm import tqdm
from huggingface_hub import HfApi


HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["COQUI_TOS_AGREED"] = "1"
api = HfApi(token=HF_TOKEN)
ZipFile("ffmpeg.zip").extractall()
st = os.stat('ffmpeg')
os.chmod('ffmpeg', st.st_mode | stat.S_IEXEC)
#Whisper
model_size = "small"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

    
def process_video(radio, video, target_language, has_closeup_face):

    run_uuid = uuid.uuid4().hex[:6]
    output_filename = f"{run_uuid}_resized_video.mp4"
    ffmpeg.input(video).output(output_filename, vf='scale=-2:720').run()

    print("Resized video to 720p.")

    video_path = output_filename

    print(f"Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        return f"Error: {video_path} does not exist."

    # Move the duration check here
    video_info = ffmpeg.probe(video_path)
    video_duration = float(video_info['streams'][0]['duration'])

    print(f"Video duration: {video_duration} seconds")

    if video_duration > 60:
        os.remove(video_path)  # Delete the resized video
        return f"Error: Video duration is {video_duration} seconds. Maximum allowed duration is 60 seconds."

    ffmpeg.input(video_path).output(f"{run_uuid}_output_audio.wav", acodec='pcm_s24le', ar=48000, map='a').run()

    #y, sr = sf.read(f"{run_uuid}_output_audio.wav")
    #y = y.astype(np.float32)
    #y_denoised = wiener(y)
    #sf.write(f"{run_uuid}_output_audio_denoised.wav", y_denoised, sr)

    #sound = AudioSegment.from_file(f"{run_uuid}_output_audio_denoised.wav", format="wav")
    #sound = sound.apply_gain(0)
    #sound = sound.low_pass_filter(3000).high_pass_filter(100)
    #sound.export(f"{run_uuid}_output_audio_processed.wav", format="wav")

    shell_command = f"ffmpeg -y -i {run_uuid}_output_audio.wav -af lowpass=3000,highpass=100 {run_uuid}_output_audio_final.wav".split(" ")
    subprocess.run([item for item in shell_command], capture_output=False, text=True, check=True)
    
    print("Attempting to transcribe with Whisper...")
    try:
        segments, info = model.transcribe(f"{run_uuid}_output_audio_final.wav", beam_size=5)
        whisper_text = " ".join(segment.text for segment in segments)
        whisper_language = info.language
        print(f"Transcription successful: {whisper_text}")
    except RuntimeError as e:
        print(f"RuntimeError encountered: {str(e)}")
        if "CUDA failed with error device-side assert triggered" in str(e):
            print("CUDA error encountered. Restarting the script...")
           
            
    language_mapping = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 'Polish': 'pl', 'Turkish': 'tr', 'Russian': 'ru', 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar', 'Chinese (Simplified)': 'zh-cn'}
    target_language_code = language_mapping[target_language]
    translator = Translator()
    translated_text = translator.translate(whisper_text, src=whisper_language, dest=target_language_code).text
    print(translated_text)

    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    # tts.to('cuda')
    # tts.tts_to_file(translated_text, speaker_wav=f"{run_uuid}_output_audio_final.wav", file_path=f"{run_uuid}_output_synth.wav", language=target_language_code)
    
    pad_top = 0
    pad_bottom = 15
    pad_left = 0
    pad_right = 0
    rescaleFactor = 1

    video_path_fix = video_path


    # Merge audio with the original video without running Wav2Lip
    cmd = f"ffmpeg -i {video_path} -i {run_uuid}_output_synth.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {run_uuid}_output_video.mp4"
    subprocess.run(cmd, shell=True)

    if not os.path.exists(f"{run_uuid}_output_video.mp4"):
        raise FileNotFoundError(f"Error: {run_uuid}_output_video.mp4 was not generated.")

    output_video_path = f"{run_uuid}_output_video.mp4"

    # Cleanup: Delete all generated files except the final output video
    files_to_delete = [
        # f"{run_uuid}_resized_video.mp4",
        # f"{run_uuid}_output_audio.wav",
        # f"{run_uuid}_output_audio_final.wav",
        # f"{run_uuid}_output_synth.wav"
    ]
    for file in files_to_delete:
        try:
            os.remove(file)
        except FileNotFoundError:
            print(f"File {file} not found for deletion.")

    return output_video_path
    
    

def main():
    video = "video.mp4"
    radio = "radio.mp3"
    target_language = "English"
    has_closeup_face = False
    process_video(radio, video, target_language, has_closeup_face)


if __name__ == "__main__":
    main()