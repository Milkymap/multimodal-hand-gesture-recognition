import cv2 

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import tensorflow as tf 

from scipy.spatial.distance import euclidean as measure 


def create_screen(name, width, height, position=None):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(name, width, height)
	if position is not None: 
		cv2.moveWindow(name, *position)

def get_nodes(landmark):
	return np.array([ [pnt.x, pnt.y] for pnt in landmark])

def normalize_nodes(nodes, scaler):
	rescaled_nodes = nodes * scaler
	return rescaled_nodes.astype('int32')

def make_adjacency_matrix(nodes):
	nb_node = len(nodes)  
	pairwise = list(it.product(nodes, nodes))	
	weighted_edges = np.array([ measure(*item) for item in pairwise ])
	return np.reshape(weighted_edges, (nb_node, nb_node)) / np.max(weighted_edges)

def get_contours(nodes):
	return cv2.boundingRect(nodes)

def draw_message_on_screen(target_screen, message, message_config, message_position):
	(tw, th), tb = cv2.getTextSize(message, **message_config)
	tx, ty = message_position
	cv2.putText(target_screen, message, (tx - tw // 2, ty + th // 2 + tb), color=(0, 255, 255), **message_config)


def create_sparse_mapper(labels):
	unique_values = np.unique(labels)
	return dict(zip(unique_values, range(len(unique_values))))

def create_model(cnn_shape, dnn_spae, nb_classes):
	cnn_input = tf.keras.Input(shape=cnn_shape, name='cnn_input')
	dnn_input = tf.keras.Input(shape=dnn_spae, name='dnn_input')

	dnn_hidden_0 = tf.keras.layers.Dense(units=128, activation='relu')(dnn_input)
	dnn_hidden_1 = tf.keras.layers.Dense(units=64, activation='relu')(dnn_hidden_0)


	cnn_hidden_0 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3))(cnn_input)
	cnn_hidden_1 = tf.keras.layers.MaxPooling2D((3, 3))(cnn_hidden_0)
	cnn_hidden_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))(cnn_hidden_1)
	cnn_hidden_3 = tf.keras.layers.MaxPooling2D((3, 3))(cnn_hidden_2)
	cnn_hidden_4 = tf.keras.layers.Flatten()(cnn_hidden_3)

	concatanator = tf.keras.layers.Concatenate()([dnn_hidden_1, cnn_hidden_4])
	normalizer = tf.keras.layers.BatchNormalization()(concatanator)

	fcn_hidden_0 = tf.keras.layers.Dense(units=64, activation='relu')(normalizer)
	fcn_hidden_1 = tf.keras.layers.Dropout(0.5)(fcn_hidden_0)
	fcn_hidden_2 = tf.keras.layers.Dense(units=32, activation='relu')(fcn_hidden_1)
	fcn_hidden_3 = tf.keras.layers.Dropout(0.1)(fcn_hidden_2)
	fcn_hidden_4 = tf.keras.layers.Dense(units=nb_classes, activation='softmax')(fcn_hidden_3)

	return tf.keras.models.Model(
		inputs=[cnn_input, dnn_input],
		outputs=fcn_hidden_4
	)

