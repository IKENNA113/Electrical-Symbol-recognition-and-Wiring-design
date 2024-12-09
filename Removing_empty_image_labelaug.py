import os
import glob
import shutil

def move_empty_files(data_dir, empty_dir):
    os.makedirs(empty_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    for txt_file in txt_files:
        image_file = txt_file.replace('.txt', '.png')

        # Check if the TXT file is empty
        with open(txt_file, 'r') as f:
            content = f.read().strip()
            if not content:
                shutil.move(txt_file, empty_dir)
                print(f"Moved empty TXT file: {txt_file}")

                # Check if the corresponding PNG file exists, then move it
                if os.path.isfile(image_file):
                    shutil.move(image_file, empty_dir)
                    print(f"Moved corresponding PNG file: {image_file}")

def main():
    path_to_data = input("Enter the path to your dataset (where the PNG and TXT files are located): ")
    path_to_empty_folder = input("Enter the path to the folder where you want to move the empty files: ")
    
    move_empty_files(path_to_data, path_to_empty_folder)
    print("Process completed. Empty TXT files and corresponding PNG files have been moved.")

if __name__ == "__main__":
    main()

