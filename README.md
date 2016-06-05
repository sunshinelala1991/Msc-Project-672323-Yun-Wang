#This project recognizes species of bird from audio data. 
The MFCC algorithm is implemented based on the work of Gyorgy Fazekas from Queen Mary University of London.



Install all the python packages so that there is no compiling errors.

Since an already trained model is included in my submission, you can use the prediction function easily. 

1. change the following lines in webapp.py to an absolute path on your computer:
 
 
UPLOAD_FOLDER = '/Users/yunwang/myFile/unimelb/projectCode/tmp/'


2. Then run webapp.py,  choose upload new file and predict. 

3. Choose a file from webpageTest folder to get its prediction. 


However, if you want to update the prediction model, you’ll need to follow the steps below:


1 Run DownloadAudioFIles.py to download all the audio files to the “audio” folder.

2 Convert the audio files to sample rate 44100Hz, “wav” format( I used iTunes for this). The file names are changed manually to the following format:
 
                AustralianGoldenWhistler_7.wav 

where the string before “_” is the name of the bird.
Store the files into a folder called “wav”

3 Run populateDatabase.py to store features into the database. You will need a local mongodb database

4 Run webapp.py and click update prediction model on the webpage. Wait for a few seconds and the model on the server will be updated.
