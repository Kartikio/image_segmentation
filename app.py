from flask import Flask, render_template, url_for, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import os
import imageio
#----------------------------------------------------------------------
def showing_predictions(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (96, 128), method='nearest')
    image = image[tf.newaxis, ...]
    pred_mask = my_model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = tf.squeeze(pred_mask, axis=0)
    pred_mask = tf.cast(pred_mask, dtype=tf.uint8)
    pred_mask = tf.keras.utils.array_to_img(pred_mask)
    return np.array(pred_mask)
#--------------------------------------------------------------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///videos.db'
db = SQLAlchemy(app)

base_dir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
my_model = tf.keras.models.load_model(base_dir + '/models/model_v2.h5',  compile=False)
UPLOAD_FOLDER = os.path.join(base_dir, 'static/saved_videos/').replace("\\", "/")


class Segmentation(db.Model):
    id = db.Column(db.Integer, primary_key = True, autoincrement = True)
    video_file = db.Column(db.String(100), nullable = False)

    def __repr__(self):
        return f'<Video {self.id}'


with app.app_context():
    db.create_all()

@app.route('/')
def index():
    videos = Segmentation.query.order_by(Segmentation.id).all()
    return render_template('index.html', videos = videos)

@app.route('/', methods = ['POST', 'GET'])
def video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file_obj = request.files['file']

        if file_obj.filename == '':
            flash('No video selected for uploading.')
            return redirect(request.url)
        else:
            video_file = secure_filename(file_obj.filename)
            video_file_path = UPLOAD_FOLDER + video_file
            file_obj.save(video_file_path)


            cap = cv2.VideoCapture(video_file_path)
            writer = imageio.get_writer(base_dir + '/static/segmented_videos/' + video_file, fps = 15 )
            while True:
                ret, frame = cap.read()
                if ret == False:
                    break
                cv2.imshow('Image', frame)
                seg_img = showing_predictions(frame)
                seg_img = cv2.resize(seg_img, (0, 0), fx=5, fy=5)
                seg_img = cv2.applyColorMap(seg_img, cv2.COLORMAP_WINTER)
                writer.append_data(seg_img)
                # cv2.imshow('Segmented Image', seg_img)

                if cv2.waitKey(1) == ord('x'):
                    break
            writer.close()
            cap.release()
            cv2.destroyAllWindows()


            new_obj = Segmentation(video_file = str(video_file))
            try:
                db.session.add(new_obj)
                db.session.commit()
            except:
                flash('Sorry the video could not be uploaded. Try Again.')
                return redirect('/')
            
            flash('Video uploaded successfully.')
            videos = Segmentation.query.order_by(Segmentation.id).all()
            return render_template('index.html', saved_video_file = video_file)


@app.route('/display_saved_video/<vid_file>')
def display_saved_video(vid_file):
    videos = Segmentation.query.order_by(Segmentation.id).all()
    return render_template('index.html', videos = videos, saved_video_file = vid_file)

@app.route('/display_segmented_video/<vid_file>')
def display_segmented_video(vid_file):
    videos = Segmentation.query.order_by(Segmentation.id).all()
    return render_template('index.html', videos = videos, segmented_video_file = vid_file)


@app.route('/display_sav_video/<video_file>')
def display_sav_video(video_file):
    print(video_file)
    return redirect(url_for('static', filename = 'saved_videos/' + video_file), code = 301)

@app.route('/display_seg__video/<video_file>')
def display_seg_video(video_file):
    return redirect(url_for('static', filename = 'segmented_videos/' + video_file), code = 301)


@app.route('/delete/<int:id>')
def delete(id):
    video_file = Segmentation.query.get_or_404(id)
    try:
        video_path = base_dir + '/static/saved_videos/' + video_file.video_file
        segmented_path = base_dir + '/static/segmented_videos/' + video_file.video_file

        os.remove(video_path)
        os.remove(segmented_path)

        db.session.delete(video_file)
        db.session.commit()
        flash('Successfully deleted video.')
        return redirect('/')
    except:
        flash('Sorry that video does not exit.')
        return redirect('/')
        

if __name__ == "__main__":
    app.secret_key = 'My_key'
    app.run(debug = True)