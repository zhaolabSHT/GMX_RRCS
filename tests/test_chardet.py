
import os
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def main():
    for infile in os.listdir('../packages'):
        file_path = os.path.join('../packages', infile)
        if os.path.isfile(file_path):
            encoding = detect_encoding(file_path)
            print(f"The encoding of the file is: {encoding}")

if __name__ == "__main__":
    main()