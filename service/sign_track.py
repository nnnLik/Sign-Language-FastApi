from aiortc.contrib.media import MediaStreamTrack
import csv
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from av import VideoFrame
import json

from aiortc import MediaStreamTrack

from model import KeyPointClassifier, PointHistoryClassifier

from .ml import (
    append_word, pre_process_landmark,
    find_middle, calc_landmark_list,
    pre_process_point_history, draw_landmarks,
    solve
)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )


class Sign(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, length=16):
        super().__init__()

        self.track = track
        self.channel = None

        self.word = ''
        self.sentence = []
        self.last_sentence = []
        self.history_length = length

        self.right_sign_id = []
        self.left_sign_id = []

        self.point_history_right = deque(maxlen=length)
        self.point_history_left = deque(maxlen=length)

        self.finger_gesture_history_right = deque(maxlen=length)
        self.finger_gesture_history_left = deque(maxlen=length)

    def get_id(self, landmark_list, debug_image, point_history, 
            keypoint_classifier, history_length,
            point_history_classifier, finger_gesture_history, side="right"):
        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = pre_process_landmark(
            landmark_list)
        if side == "left":
            b = [0] * len(pre_processed_landmark_list)
            for i, el in enumerate(pre_processed_landmark_list):
                b[i] = -el if i % 2 == 0 else el
            pre_processed_landmark_list = b
        pre_processed_point_history_list = pre_process_point_history(
            debug_image, point_history)
        # Hand sign classification
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        # Point gesture
        point_history.append(landmark_list[0])
        
        # Finger gesture classification
        finger_gesture_id = 0
        point_history_len = len(pre_processed_point_history_list)
        if point_history_len == (history_length * 2):
            finger_gesture_id = point_history_classifier(
                pre_processed_point_history_list)
        
        # Calculates the gesture IDs in the latest detection
        finger_gesture_history.append(finger_gesture_id)
        most_common_fg_id = Counter(
            finger_gesture_history).most_common()
        hand_gest_id = most_common_fg_id[0][0]

        return hand_sign_id, hand_gest_id

    async def recv(self):

        frame = await self.track.recv()
        image = frame.to_ndarray(format="bgr24")

        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                    encoding='utf-8-sig') as f:
                keypoint_classifier_labels = csv.reader(f)
                keypoint_classifier_labels = [
                    row[0] for row in keypoint_classifier_labels
                ]
        with open(
                    'model/point_history_classifier/point_history_classifier_label.csv',
                    encoding='utf-8-sig') as f:
                point_history_classifier_labels = csv.reader(f)
                point_history_classifier_labels = [
                    row[0] for row in point_history_classifier_labels
                ]

        # hist_gest = []
        # hist_id = []
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = cv.flip(image, 1) 
        results = holistic.process(image)
       
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        try:
            pose_landmarks = results.pose_landmarks.landmark
            left_shoulder = [pose_landmarks[11].x, pose_landmarks[11].y]
            right_shoulder = [pose_landmarks[12].x, pose_landmarks[12].y]
            right_eye= [pose_landmarks[4].x, pose_landmarks[4].y]
            left_eye= [pose_landmarks[1].x, pose_landmarks[1].y]
            nose =  [pose_landmarks[0].x, pose_landmarks[0].y]
            mid_shoulders = find_middle(left_shoulder, right_shoulder)
            right_hip = [pose_landmarks[24].x, pose_landmarks[24].y]
            left_hip = [pose_landmarks[23].x, pose_landmarks[23].y]
            mid_hip=find_middle(left_hip,right_hip)
            middle=find_middle(mid_hip,mid_shoulders)
            middle_eye=find_middle(left_eye,right_eye)

            handedness = "right"
            #  ####################################################################
            #  Right hand
            if results.left_hand_landmarks is not None:
                
                hand_landmarks = results.left_hand_landmarks.landmark
                right_wrist = [hand_landmarks[0].x, hand_landmarks[0].y]
                right_pointer = [hand_landmarks[8].x, hand_landmarks[8].y]
                # Bounding box calculation
                handedness = "right"
                
                # Landmark calculation
                landmark_list = calc_landmark_list(image, hand_landmarks)

                right_hand_sign_id, right_hand_gest_id = self.get_id(
                            landmark_list, image, self.point_history_right, 
                            keypoint_classifier, self.history_length,
                            point_history_classifier, self.finger_gesture_history_right,
                            side=handedness
                )
                image = draw_landmarks(image, landmark_list)
                
            else:
                self.point_history_right.append([0, 0])
                right_hand_sign_id, right_hand_gest_id = -1, -1
            ############################################################################
            # Left hand
            if results.right_hand_landmarks is not None:
                hand_landmarks = results.right_hand_landmarks.landmark
                left_wrist = [hand_landmarks[0].x, hand_landmarks[0].y]
                # Bounding box calculation
                handedness = "left"
                landmark_list = calc_landmark_list(image, hand_landmarks)

                left_hand_sign_id, left_hand_gest_id = self.get_id(
                            landmark_list, image, self.point_history_left, 
                            keypoint_classifier, self.history_length,
                            point_history_classifier, self.finger_gesture_history_left,
                            side=handedness
                )
                image = draw_landmarks(image, landmark_list)
            
            else:
                self.point_history_left.append([0, 0])
                left_hand_sign_id, left_hand_gest_id = -1, -1  

            if len(self.sentence) > 2:
                self.sentence = self.sentence[-2:]
            
            self.right_sign_id.append(right_hand_sign_id)
            if len( self.right_sign_id) > 5:
                     self.right_sign_id = self.right_sign_id[-5:]

            if len(set(self.right_sign_id)) == 1:
                  right_hand_sign_id = self.right_sign_id[0]
            elif len(set(self.right_sign_id)) > 2:
                right_hand_sign_id = -1
            else:
                right_hand_sign_id =  Counter(self.right_sign_id).most_common()[0][0]


            self.left_sign_id.append(left_hand_sign_id)
            if len(self.left_sign_id) > 5:
                    self.left_sign_id=self.left_sign_id[-5:]

            if len(set(self.left_sign_id)) == 1:
                left_hand_gest_id = self.left_sign_id[0]
            elif len(set(self.left_sign_id)) > 2:
                left_hand_sign_id = -1
            else:
                left_hand_sign_id =  Counter(self.left_sign_id).most_common()[0][0]
                    
            if right_hand_sign_id in [7, 8]  and right_hand_gest_id in [1, 8, 0, 4, 2] \
                and solve(middle, left_shoulder, right_wrist, "left"):
                    append_word(self.sentence, 'pain')                
            elif left_hand_sign_id == 7  and left_hand_gest_id in [5, 3, 0, 2]\
                and solve(middle, right_shoulder, left_wrist, "right"):
                append_word(self.sentence, 'pain')                       
            elif right_hand_sign_id in [6, 4]  and right_hand_gest_id not in [4, 7, 8]\
                and solve(left_shoulder, middle_eye, right_wrist, "right") and \
                left_hand_sign_id == -1:
                append_word(self.sentence, 'hello')                
            elif left_hand_sign_id in [6, 4]  and left_hand_gest_id not in [4, 8]\
                and solve(right_shoulder, middle_eye, left_wrist, "left"):
                append_word(self.sentence, 'hello')           
            elif right_hand_sign_id == 9 and right_hand_gest_id in [0, 2, 6, 3]\
                and right_pointer[1] < right_shoulder[1] and right_pointer[1] > nose[1]\
                    and left_hand_sign_id == 9:
                append_word(self.sentence, 'thanks')

            elif right_hand_sign_id == 11\
                and left_hand_sign_id == 11 and right_wrist[1] > right_shoulder[1]\
                    and left_wrist[1] > left_shoulder[1]:
                append_word(self.sentence,  'depression')
            elif right_hand_sign_id == 12 and left_hand_sign_id == 12\
                and not solve(middle, left_shoulder, right_wrist, "left"):
                append_word(self.sentence,  'health')
            elif right_hand_sign_id == 10 and right_hand_gest_id in [2, 7, 8]\
                and solve(middle, left_shoulder, right_wrist, "left"):
                append_word(self.sentence,  'depression')
            elif (right_hand_sign_id == 1 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]) \
                    or (left_hand_sign_id == 1 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence,  'a')
            elif right_hand_sign_id == 0 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 0 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence,  'b')
            elif right_hand_sign_id == 3 and right_hand_gest_id == 0\
            and right_wrist[1] > right_shoulder[1]\
                or (left_hand_sign_id == 3 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence,  'c')
            elif right_hand_sign_id == 4 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 4 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence,  'e')
            elif right_hand_sign_id == 5 and right_hand_gest_id == 0\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 5 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence,  'i')
            elif right_hand_sign_id == 3 and right_hand_gest_id == 2\
                and right_wrist[1] > right_shoulder[1]\
                    or (left_hand_sign_id == 3 and left_hand_gest_id == 0\
                and left_wrist[1] > left_shoulder[1]):
                append_word(self.sentence,  'ch')
            else:
                pass

        except:
            pass

        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base


        print(f'{self.sentence} -- {self.last_sentence}')
        # log
        if self.channel and self.sentence:
            if self.last_sentence:
                if self.sentence[-1] != self.last_sentence[-1]:
                    self.channel.send(json.dumps({'word': self.sentence[-1]}))
            else:
                self.channel.send(json.dumps({'word': self.sentence[-1]}))

            self.last_sentence.append(self.sentence[-1])
            if len(self.last_sentence) >= 1:
                self.last_sentence = self.last_sentence[-1:]