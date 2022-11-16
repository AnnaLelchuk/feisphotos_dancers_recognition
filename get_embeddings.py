import matplotlib.pyplot as plt
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import conf as CFG
from keras_vggface.utils import decode_predictions

class Embeddings:
    def __init__(self, pic_path):
        """ Receives a pic path to initialize, reads to np.array, rotates 90deg counterclockwise.
        Returns np.array of the pic """
        img = plt.imread(pic_path)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.img = img

    def main(self):
        # get face array
        # self.face_array = self.crop_face(self.img)
        self.get_face_array()
        self.face_emb = self.get_face_embedding([self.face_array])

        # get body array
        # self.net = self.configure_yolo()
        # self.body_array = self.crop_body()
        self.get_body_array()
        self.body_emb = self.get_body_embedding(self.body_array)
        return [self.face_emb, self.body_emb]

    def get_face_array(self):
        self.face_array = self.crop_face(self.img)
        self.face_emb = self.get_face_embedding([self.face_array])
        return self.face_array

    def get_body_array(self):
        self.net = self.configure_yolo()
        self.body_array = self.crop_body()
        return self.body_array

    def crop_face(self, img, required_size=(224, 224)):
        """
        Uses MTCNN to detect face from a photo.
        Returns cropped face
        """
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(self.img)

        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = img[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        self.face_array = asarray(image)

        return self.face_array

    def get_face_embedding(self, face_array):
        """ return VGGFace representation """
        '''Calculates face embeddings for a list of photo files'''
        # convert into an array of samples
        face_array = asarray(face_array, 'float32')
        # prepare the face for the model, e.g. center pixels
        face_array = preprocess_input(face_array, version=2)
        # create a vggface model
        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        # create embedding
        self.face_emb = model.predict(face_array)
        return self.face_emb

    def crop_body(self, required_size=(224, 224)):
        """
        Uses Yolo to detect a person. Crops upper body from the person.
        Returns cropped upper body as a numpy array.
        """
        box, confidence = self.detect_person()
        if len(box) != 0:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            # extract person from a photo
            person = self.img[y:y + h, x:x + w]

            # crop upper body based on avg body and legs percentage of a person, resize to VGG size
            self.upper_body = person[round(person.shape[0] * CFG.HEAD_PERC): round(person.shape[0] * (1 - CFG.LEGS_PERC)), :]
            self.upper_body = cv2.resize(self.upper_body, required_size)

        return self.upper_body



    def output_coordinates_to_box_coordinates(self, cx, cy, w, h, img_w, img_h):
        """Utility function for Yolo body detection"""
        abs_x = int((cx - w / 2) * img_w)
        abs_y = int((cy - h / 2) * img_h)
        abs_w = int(w * img_w)
        abs_h = int(h * img_h)
        return abs_x, abs_y, abs_w, abs_h

    def detect_person(self, conf_thresh=.5, nms_thresh=.4):
        """"""
        img_h, img_w = self.img.shape[:2]

        # getting a blob
        blob = cv2.dnn.blobFromImage(self.img, 1 / 255, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # getting predictions
        output_names = list(self.net.getUnconnectedOutLayersNames())[-3:]
        large, medium, small = self.net.forward(output_names)
        all_outputs = np.vstack((large, medium, small))  # all the pedictions
        objs = all_outputs[all_outputs[:, 4] >= conf_thresh]  # filtering just those with higher confidence and persons only
        objs = objs[np.any(objs[:, 5:] == 0, axis=1)]

        # getting boxes
        boxes = []
        for row in objs:
            cx, cy, w, h = row[:4]
            x_value, y_value, width, height = self.output_coordinates_to_box_coordinates(cx, cy, w, h, img_w, img_h)
            boxes.append([x_value, y_value, width, height])

        confidences = list(map(float, objs[:, 4]))

        # applying NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

        boxes = np.array(boxes)[[indices]]
        confidences = np.array(confidences)[[indices]]

        # if there are several boxes, chose 'widest' (that would probably give a person in the front)
        if len(indices) == 1:
            self.box = boxes[0]
            self.confidence = confidences[0]

        elif len(indices) > 1:

            widths = np.array([box[2] for box in boxes])
            final_idx = np.argmax(widths)
            self.box = boxes[final_idx]
            self.confidence = confidences[final_idx]
        return self.box, self.confidence

    def get_body_embedding(self, upper_body):
        """ return VGG16 representation  """
        # crop the body YOLO
        # pass the input the face to VGG16 to get embeddings
        # return VGGFace

        # UPDATE WITH AN ACTUAL CODE!
        self.body_emb = np.random.randint(2, size=10)
        return self.body_emb

    def configure_yolo(self):

        # configuring the net:
        self.net = cv2.dnn.readNetFromDarknet(CFG.YOLO_CFG, CFG.YOLO_WEIGHTS)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        return self.net

    def body_embedding_simease(self, pic):
        """ return Simease representation  """
        pass # TBD
##############################
# # API
# # initialise the object with a path
# emb = Embeddings(r"C:\Users\lelchuk\Desktop\ITC_course\810.Project_2\Great_Britain_Championships_2021\solo\U16_Girls\highres\110\_U164470.JPG")
# # to get face array:
# face_array = emb.get_face_array()
#
# # to get body array:
# face_array = emb.get_body_array()
#
# # to get face embedding:
# face_emb = emb.get_face_embedding(face_array)

