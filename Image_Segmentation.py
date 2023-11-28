import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import os
import imageio


base_dir = os.getcwd()
my_model = tf.keras.models.load_model(os.path.join(base_dir, 'models/model_v2.h5'), compile=False)


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



cap = cv2.VideoCapture(os.path.join(base_dir, 'static/saved_videos/video.avi'))
# fourcc = cv2.VideoWriter_fourcc(*'H264')
# fps = 10
# frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter(base_dir + '/static/segmented_videos/' + 'istockphoto-1249558755-640_adpp_is.mp4', fourcc, fps, frame_size)
# writer = imageio.get_writer(base_dir + '/static/segmented_videos/' + 'istockphoto-1249558755-640_adpp_is.mp4', fps = 15)
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imshow('Image', frame)
    seg_img = showing_predictions(frame)
    seg_img = cv2.resize(seg_img, (0, 0), fx=5, fy=5)
    seg_img = cv2.applyColorMap(seg_img, cv2.COLORMAP_WINTER)
    # out.write(seg_img)
    # writer.append_data(seg_img)
    cv2.imshow('Segmented Image', seg_img)

    if cv2.waitKey(1) == ord('x'):
        break
# out.release()
# writer.close()
cap.release()
cv2.destroyAllWindows()





# def grab_frame(cap):
#     ret, frame = cap.read()
#     print('Inside Grab func')
#     if ret == False:
#         print('Inside ret = False')
#         return []
#     cv2.imshow('Image', frame)
#     seg_img = showing_predictions(frame)
#     seg_img = cv2.resize(seg_img, (0, 0), fx=5, fy=5)

#     if cv2.waitKey(1) == ord('x'):
#         cap.release()
#         cv2.destroyAllWindows()
#         return []
#     return seg_img


# fig, ax = plt.subplots()
# plt.grid(False)
# plt.axis('off')
# plt.rcParams['figure.autolayout'] = True

# im = ax.imshow(grab_frame(cap))


# def update(frame):
#     im.set_data(grab_frame(cap))


# anim = animation.FuncAnimation(plt.gcf(), update, fargs = (cap) , interval=200, cache_frame_data = False)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=5)
# anim.save(os.path.join(base_dir, 'static/segmented_videos/segmented_video.avi') , writer = writer)
# try:
#     anim.save(os.path.join(base_dir, 'static/segmented_videos/segmented_video.avi') , writer = writer)
# except TypeError:
#     print('FUCKKKKKKKKKKKKKKKKKK')
#     anim.event_source.stop()
