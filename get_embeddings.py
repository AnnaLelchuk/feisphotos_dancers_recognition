import matplotlib.pyplot as plt
import numpy as np
import cv2
from mtcnn_cv2 import MTCNN
from numpy import asarray
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import conf as CFG
import os
from tqdm import tqdm
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import gdown

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Embeddings:
    """Gets embeddings in the following way:
    1. Receives a path where all the photos are
    2. Creates a list of pathes to each of the photos and converts them to np array
    3. Crops body from each photo (with Yolo) and saves that array
    4. Crops face from each detected body (with MTCNN) and saves that array
    5. Gets face embedding from cropped face with VGGFace
    6. Gets body embedding from cropped body with VGG-16
    Returns: self.body_arrays, self.face_arrays, self.face_emb, self.body_emb
    """

    def __init__(self, source_path):
        """ Initiated with path to all the photos that need to be sorted"""
        photo_names = os.listdir(source_path)
        self.img_paths = [source_path + '/' + photo for photo in photo_names]
        self.img_arrays = [cv2.rotate(plt.imread(path), cv2.ROTATE_90_COUNTERCLOCKWISE) for path in self.img_paths]
        # true_labels = [photo[:3] for photo in photo_names]

    def main(self):
        # configuring the yolo_net.
        # download Yolo weight to the folder
        cur_dir = os.getcwd()
        if r'yolo_model_weights' not in os.listdir(cur_dir):
            print('[INFO] DOWNLOADING YOLO FILES..')
            yolo_dir = 'yolo_model_weights'
            os.mkdir(yolo_dir)
            gdown.download(id=CFG.YOLO_WEIGHTS_DWNLD_ID, output=yolo_dir + '/yolov3.weights', quiet=False)
            gdown.download(id=CFG.YOLO_CFG_DWNLD_ID, output=yolo_dir + '/yolov3.cfg', quiet=False)
            print('[INFO] DOWNLOAD COMPLETE')

        yolo_model = cv2.dnn.readNetFromDarknet(CFG.YOLO_CFG, CFG.YOLO_WEIGHTS)
        yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # crop the person from the original photos
        print('\n[INFO] Detecting dancers from images..')
        self.body_arrays = [self.crop_person(img, yolo_model) for img in tqdm(self.img_arrays)]

        # configure MTCNN
        detector = MTCNN()
        # crop the person from the person
        print('\n[INFO] Detecting faces from images..')
        self.face_arrays = [self.crop_face(b, detector) if b is not None else None for b in tqdm(self.body_arrays)]

        # get face embedding
        vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        print('\n[INFO] Getting face embeddings..')
        self.face_emb = [self.get_face_embedding([f], vggface_model) if f is not None else None for f in
                         self.face_arrays]

        # get body embeddings
        # load model and remove the output layer
        print('\n[INFO] Getting body embeddings..')
        try:
            vgg_model = VGG16()
            vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
            self.body_emb = [self.get_body_embedding_vgg16([b][0], vgg_model) if b is not None else None for b in
                             self.body_arrays]
        except ModuleNotFoundError:
            print('Refer to README file for a fix. Sorry, have to exit now.')
            sys.exit()

        return self.body_arrays, self.face_arrays, self.face_emb, self.body_emb

    def output_coordinates_to_box_coordinates(self, cx, cy, w, h, img_w, img_h):
        """Utility function for body detection"""
        abs_x = int((cx - w / 2) * img_w)
        abs_y = int((cy - h / 2) * img_h)
        abs_w = int(w * img_w)
        abs_h = int(h * img_h)
        return abs_x, abs_y, abs_w, abs_h

    def crop_person(self, img, net, conf_thresh=.05, nms_thresh=.4):
        """Crops a person out pf the photo.
        Returns np array"""
        img_h, img_w = img.shape[:2]

        # getting a blob
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # getting predictions
        output_names = list(net.getUnconnectedOutLayersNames())[-3:]
        large, medium, small = net.forward(output_names)
        all_outputs = np.vstack((large, medium, small))  # all the predictions
        objs = all_outputs[all_outputs[:, 4] >= .05]  # filtering just those with higher confidence

        #  getting boxes and confidences
        boxes = []
        for row in objs:
            cx, cy, w, h = row[:4]
            x_value, y_value, width, height = self.output_coordinates_to_box_coordinates(cx, cy, w, h, img_w, img_h)
            boxes.append([x_value, y_value, width, height])
        confidences = list(map(float, objs[:, 4]))

        # applying NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        if len(indices) >= 1:

            boxes = np.array(boxes)[indices]
            confidences = np.array(confidences)[indices]

            # selecting one box out of all predicted
            # remove boxes by the edges (15 pixels or less from the edge)
            if len(boxes) > 1:
                x = np.array([box[0] for box in boxes])
                w = np.array([box[2] for box in boxes])
                x_good = np.where(x > 15)
                w_good = np.where(x + w < img.shape[1] - 15)
                ind_choice = np.intersect1d(x_good, w_good)
                boxes = boxes[ind_choice]
                confidences = confidences[ind_choice]

                # if still more than one remains, keep only with teh max height
                if len(boxes) > 1:
                    h = [box[3] for box in boxes]
                    final = np.argmax(h)
                    boxes = boxes[final].reshape(1, 4)
                    confidences = confidences[final]

            x = boxes[0][0]
            y = boxes[0][1]
            w = boxes[0][2]
            h = boxes[0][3]

            person = img[max(y, 0): min(y + h, img.shape[0]), max(x, 0):min(x + w, img.shape[1])]

        else:
            person = None

        return person

    def crop_face(self, img_array, detector, required_size=(224, 224)):
        face_arrays = []
        results = detector.detect_faces(img_array)
        if results != []:
            # extract the bounding box from the first face
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = img_array[y1:y2, x1:x2]
            # resize pixels to the model size
            face = Image.fromarray(face)
            face = face.resize(required_size)
            face_array = asarray(face)

        else:
            face_array = None

        return face_array

    def get_face_embedding(self, face_arrays, model):
        # convert into an array of samples
        face_arrays = asarray(face_arrays, 'float32')
        # prepare the face for the model, e.g. center pixels
        face_arrays = utils.preprocess_input(face_arrays, version=2)
        # create embedding
        face_emb = model.predict(face_arrays)
        return face_emb

    def get_body_embedding_vgg16(self, body_array, model):
        # prepare image for model
        body_array = cv2.resize(body_array, (224, 224))
        body_array = body_array.reshape(1, 224, 224, 3)
        body_array = preprocess_input(body_array)

        # get the feature vector
        body_emb = model.predict(body_array, use_multiprocessing=True)
        return body_emb

##########################################################
# API
# emb = Embeddings(r"C:\Users\lelchuk\Desktop\ITC_course\810.Project_2\Test_folder")
#
# body, face, face_emb, body_emb = emb.main()
