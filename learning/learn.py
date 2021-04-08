import cv2
import argparse 
import pickle as pkl

import numpy as np 

from utilities.utils import create_model, create_sparse_mapper 


if __name__ == '__main__':
	print(' ... [learning] ... ')
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', help='path to source data', required=False, default='dump/features.pkl')
	parser.add_argument('--storage', help='path to model storage', required=False, default='models/agent.h5')
	args_map = vars(parser.parse_args())

	with open(args_map['source'], 'rb') as file_pointer:
		training_data = pkl.load(file_pointer)
		cnn_input = [ item['cnn_input'] for item in training_data]
		dnn_input = [ item['dnn_input'] for item in training_data]
		label = [ item['label'] for item in training_data]

		batch_cnn_input = np.expand_dims(np.stack(cnn_input), axis=-1) / 255.0 
		batch_dnn_input = np.vstack(dnn_input)

		print(batch_cnn_input.shape, batch_dnn_input.shape)

		sparse_mapper = create_sparse_mapper(label)
		sparsed_label = list(map(lambda e: sparse_mapper[e], label))
		prepared_label = np.vstack(sparsed_label)


		print(prepared_label)

		model = create_model(batch_cnn_input.shape[1:], batch_dnn_input.shape[1:], len(sparse_mapper))
		model.summary()

		model.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)

		model.fit(
			x={
				'cnn_input':batch_cnn_input, 
				'dnn_input':batch_dnn_input,
			}, 
			y=prepared_label,
			epochs=10, 
			batch_size=16,
			verbose=1,
			shuffle=True
		)

		model.save(args_map['storage'])
