import os
from flask import Flask, request, redirect, url_for,render_template
import numpy as np
from BirdSongClassification import BirdSong
import pickle
import pymongo


# the place to store the uploaded audio files, must be absolute path
UPLOAD_FOLDER = '/Users/yunwang/myFile/unimelb/projectCode/tmp/'
# the file formats allowed
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

'''
The method used to tell if the file uploaded is in the allowed format
takes file name as input
'''
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


'''
The page where allows users to upload new audio file, update machine learning model,
visualise the location on the map
'''
@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filename_split=filename.split("_")
            lat,lng=filename_split[-3:-1]
            print lat,lng

            birdsong=BirdSong()

            #load stored model and means and inverse standard deviations
            with open('objs.pickle') as f:
                gmm,means,invstds = pickle.load(f)


            #calculate features for the uploaded file and predict
            mfcc_feat1=birdsong.file_to_features(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            mfcc_feat=(mfcc_feat1-means)*invstds
            bird_name,confidence=birdsong.predict_one_bird(mfcc_feat,gmm)


            # if confident enough, add feature to the database. The threshold is set manually as 0.7
            if confidence>0.7:
                features = mfcc_feat1.tolist()
                connection = pymongo.MongoClient("mongodb://localhost")
                # get a handle to the bird database
                db = connection.bird
                birdFeature = db.birdFeature
                count=birdFeature.find().count()+1
                a_bird = {'_id': bird_name+'_'+str(count), 'feature': features}
                print('Inserted',bird_name+'_'+str(count))
                try:
                    birdFeature.insert_one(a_bird)

                except Exception as e:
                    print "Unexpected error:", type(e), e


            confidence="{0:.2f}".format(confidence)

            return render_template('bird.html',name=bird_name,confidence=confidence,lat=lat,lng=lng)

    return render_template('bird.html')





'''
This is the url used to update the stored model using the most up-to-date data
'''

@app.route('/update', methods=['GET', 'POST'])
def update():
    print 'this is the output of button click'

    birdsong=BirdSong()

    dic=birdsong.get_feature_dic_mongodb("mongodb://localhost")

    allconcat = np.vstack((dic.values()))

    means = np.mean(allconcat, 0)
    invstds = np.std(allconcat, 0)

    for i, val in enumerate(invstds):
        if val == 0.0:
            invstds[i] = 1.0
        else:
            invstds[i] = 1.0 / val

    normedTrain=birdsong.normed_features(dic,means,invstds)
    gmm =birdsong.train_the_model(normedTrain)

    print(len(gmm) ,'len gmm')

    list_to_save=[gmm,means,invstds]

    with open('objs.pickle', 'w') as f:
        pickle.dump(list_to_save, f)

    return redirect(url_for('index'))





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)
