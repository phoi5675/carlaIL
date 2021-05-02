#!/usr/bin/env python

import tensorflow as tf

import os
from imitation_learning_network import load_imitation_learning_network


class Network:
    def __init__(self, model_name, model_dir='/model/', memory_fraction=0.25):
        self.image_size = (88, 200, 3)  # 아마 [세로, 가로, 차원(RGB)] 인듯?
        # load tf network model
        # TODO dropout_vec 제일 뒤에 있는 값은 학습 파일과 동일해야 함
        self.dropout_vec = [1.0] * 8 + [0.8] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
        config_gpu = tf.ConfigProto()  # tf 설정 프로토콜인듯?

        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

        config_gpu.gpu_options.visible_device_list = '0'  # GPU >= 2 일 때, 첫 번째 GPU만 사용

        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction  # memory_fraction % 만큼만 gpu vram 사용

        self.sess = tf.Session(config=config_gpu)  # 작업을 위한 session 선언

        with tf.device('/gpu:0'):  # 수동으로 device 배치 / default : '/gpu:0'
            # tf.placeholder(dtype, shape, name) 형태로 shape에 데이터를 parameter로 전달
            self.input_images = tf.placeholder("float", shape=[None, self.image_size[0],
                                                               self.image_size[1],
                                                               self.image_size[2]],
                                               name="input_image")

            self.input_data = []

            # input control 종류가 4가지니까 [None, 4]로 지정?
            self.input_data.append(tf.placeholder(tf.float32,
                                                  shape=[None, 4], name="input_control"))

            self.input_data.append(tf.placeholder(tf.float32,
                                                  shape=[None, 1], name="input_speed"))

            # dropout vector 값. 아마 신경망이랑 관련 있는듯
            self.dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope('Network'):  # 아래의 network_tensor 가 Network 아래의 신경망임을 명시 -> Network/network_tensor 의 형태
            # 여기에 있는 load_imitation_learning_network 가 신경망 자체
            self.network_tensor = load_imitation_learning_network(self.input_images,
                                                                  self.input_data,
                                                                  self.image_size, self.dout)

        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + model_dir

        # 그래프 초기화
        # tf.reset_default_graph()

        # 변수 초기화 -> 작업 전 명시적으로 수행 / session 실행
        self.sess.run(tf.global_variables_initializer())
        self.load_model()

    def load_model(self):
        # 이전에 학습한 결과 로드
        variables_to_restore = tf.global_variables()

        # 모델, 파라미터 저장
        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        # checkpoint 가 존재하는 경우 로드
        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        return ckpt
