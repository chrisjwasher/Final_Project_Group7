import os
import subprocess

def convert_all_mp4_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.mp4'):
            mp4_file = os.path.join(input_folder, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_file = os.path.join(output_folder, wav_filename)
            print(f"Converting {mp4_file} to {wav_file}...")

            try:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", mp4_file,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    wav_file
                ], check=True)
            except subprocess.CalledProcessError as e:
                # This will catch ffmpeg errors (non-zero exit status)
                print(f"Error converting {mp4_file}: {e}")
            except FileNotFoundError:
                # This will catch if ffmpeg is not found
                print("ffmpeg not found. Please install ffmpeg and ensure it's on your PATH.")

    print("Conversion complete!")

if __name__ == "__main__":
    input_folder = "/home/ubuntu/DL_Lectures/Final_Project/Data/MELD.Raw/train_splits"
    output_folder = "/home/ubuntu/DL_Lectures/Final_Project/Data/MELD.Raw/train_audio"
    convert_all_mp4_to_wav(input_folder, output_folder)
