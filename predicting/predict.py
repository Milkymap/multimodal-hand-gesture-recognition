import cv2 
import argparse 

import pickle as pkl 
import mediapipe as mp 


import numpy as np 
import operator as op 
import itertools as it, functools as ft 

from utilities.utils import * 

def grab(model_storage):
	try: 
		W, H = 640, 480 
		screen_0 = 'screen_0'
		screen_1 = 'screen_1'
		screen_2 = 'screen_2'
		screen_3 = 'screen_3'


		create_screen(screen_0, W, H, (10, 10))
		create_screen(screen_1, W, H, (1000, 10))
		create_screen(screen_2, W, H, (1000, 600))
		create_screen(screen_3, W, H, (10,   600))

		sigma = 'aze'
		sorted_sigma = sorted(list(sigma), key=ord)
		messages = {
			'e': 'Thumbs-UP', 
			'a': 'Thumbs-Down', 
			'z': 'Reference'
		}
		message_config = {
			'fontFace': cv2.FONT_HERSHEY_SIMPLEX, 
			'fontScale': 3, 
			'thickness': 7
		}
		message_position_reference = (W // 2, H // 2) 

		scaler = np.array([W, H])
		
		predictor = tf.keras.models.load_model(model_storage, compile=False)

		mp_builder = mp.solutions.hands 
		mp_drawing = mp.solutions.drawing_utils

		mp_builder_config = {
			'max_num_hands': 2, 
			'min_tracking_confidence': 0.7,
			'min_detection_confidence': 0.7
		}

		draw_spec_0 = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=3)
		draw_spec_1 = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5)
		

		with mp_builder.Hands(**mp_builder_config) as detector: 
			skeleton = np.zeros((H, W, 3), dtype=np.uint8)
			sign = np.zeros_like(skeleton)

			capture = cv2.VideoCapture(0)
			keep_processing = True 
			while keep_processing:
				key_code = cv2.waitKey(25) & 0xFF 
				keep_processing = key_code != 27
				capture_status, bgr_frame = capture.read()
				if capture_status and keep_processing:
					rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
					response = detector.process(rgb_frame)
					output = response.multi_hand_landmarks
					if output:
						hand = output[0]
						nodes = get_nodes(hand.landmark)
						rescaled_nodes = normalize_nodes(nodes, scaler)
						matrix = make_adjacency_matrix(rescaled_nodes)
						resized_matrix = cv2.resize((matrix * 255).astype('uint8'), (640, 480))
						x, y, w, h = get_contours(rescaled_nodes)
						mp_drawing.draw_landmarks(skeleton, hand, mp_builder.HAND_CONNECTIONS, draw_spec_0, draw_spec_1)

						if key_code == 255:
							hand_roi = skeleton[y-10:y+h+10, x-10:x+w+10, :].copy()
							resized_hand_roi = cv2.resize(hand_roi, (32, 32), interpolation=cv2.INTER_CUBIC)
							gray_resized_hand_roi = cv2.cvtColor(resized_hand_roi, cv2.COLOR_BGR2GRAY)
							flattened_matrix = np.ravel(matrix)
							
							input_batch = {
								'cnn_input': gray_resized_hand_roi[None, :, :, None] / 255, 
								'dnn_input': flattened_matrix[None, :],		
							}

							output = np.ravel(predictor.predict(input_batch))
							candidate = np.argmax(output)

							print(output, candidate, sorted_sigma[candidate])
							draw_message_on_screen(
								sign, 
								messages[sorted_sigma[candidate]], 
								message_config, 
								message_position_reference
							)

						cv2.rectangle(skeleton, (x, y), (x + w, y + h), (0, 0, 255), 2)
						cv2.imshow(screen_1, resized_matrix)

					
					cv2.imshow(screen_2, skeleton)	
					cv2.imshow(screen_3, sign)
					cv2.imshow(screen_0, bgr_frame)
					skeleton *= 0
					sign *= 0 

			capture.release()
			cv2.destroyAllWindows()

	except KeyboardInterrupt as e:
		print(e)
	except Exception as e:
		print(e)

if __name__ == '__main__':
	print(' ... [building] ... ')
	parser = argparse.ArgumentParser()
	parser.add_argument('--storage', help='path to model storage', required=False, default='models/agent.h5')
	args_map = vars(parser.parse_args())
	
	grab(args_map['storage'])
