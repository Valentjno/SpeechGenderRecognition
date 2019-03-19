# SpeechGenderRecognition

This is project allows to recognize from a wav file the gender of who speak.

If you have Telegram you can try the output of this project contacting: **[@GenderSpeechBot](https://t.me/GenderSpeechBot)**, this bot is open source and available [here](https://github.com/Helias/Speech-Gender-Recognition-Bot).

To install the project locally, read the follows instructions.

## Requirements

- Python3

**Libraries**
- scikit-image
- pandas
- numpy
- matplotlib
- pyAudioAnalysis
- scipy
- sklearn
- hmmlearn
- simplejson
- eyed3
- pydub

You can install this requirements with
```
$ pip3 install -r requirements.txt
```

## Usage

To run the classifiers use:

```
python3 main.py -r
```

To run the classifier with a new sample.wav use:

```
python3 main.py -w path/file.wav
```

**Note**: to convert any audio file into .wav file we recommend [SoundConverter](https://soundconverter.org/)


To run the classifier with a new sample use:
```
python3 main.py -i path/file.csv
```


## Extract new csv samples

### Install R and packages

If you run Linux (debian-based) you can use the follows commands to install R and dependencies:
```
$ sudo apt install r-base
$ sudo apt install gfortran libsndfile1-dev libfftw3-dev
$ R
```

With the last command you just opened r, so you can run the follows command to install packages:
```
$ install.packages("tuneR", "seewave")
$ q()
```

Note: **q()** is to close the "r console".

### Run the script

Fill the folder **"male"** and **"female"** in **R/** with wavfile with the related gender.

Use **Rscript** to run the file **extract_feature.r** to generate a new file "my_voice.csv" (*backup it before generate the new one*).

```
Rscript extractor_feature.r
```


---


## Credits

- **[Valentjno](https://github.com/Valentjno)**  
- **[Helias](https://github.com/Helias)**
