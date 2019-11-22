from pathlib import Path

import numpy as np

from interact import interact as io
from nnlib import nnlib


class IMM(object):

    def __init__ (self, resolution=128,
                        initialize_weights=True,
                        load_weights_locally=False,
                        weights_file_root=None,

                        is_training=True,
                        tf_cpu_mode=0,
                        ):
        exec( nnlib.import_all(), locals(), globals() )


        bgr_shape = (resolution, resolution, 3)
        nf = 32

        self.detector = modelify ( IMM.DetectorFlow(nf) ) ( Input(bgr_shape) )
        self.feature_extractor = modelify ( IMM.FeatureExtractorFlow(nf) ) ( Input(bgr_shape) )

        shd = K.int_shape(self.detector.outputs[0])[1:]
        she = K.int_shape(self.feature_extractor.outputs[0])[1:]

        sh = ( shd[0], shd[1], shd[2]+she[2] )

        self.regressor     = modelify ( IMM.RegressorFlow(nf) ) ( Input(sh) )



        img_src = Input(bgr_shape, name="img_src")
        img_tgt = Input(bgr_shape, name="img_tgt")

        def xor_list(lst1, lst2):
            return  [value for value in lst1+lst2 if (value not in lst1) or (value not in lst2)  ]


        feature_extractor_updates = self.feature_extractor._updates.copy()
        feat_src = self.feature_extractor(img_src)
        feature_extractor_updates = xor_list (feature_extractor_updates, self.feature_extractor._updates )

        detector_updates = self.detector._updates.copy()
        heatmaps_tgt = self.detector(img_tgt)
        detector_updates = xor_list (detector_updates, self.detector._updates )

        def get_coord(x, other_axis, axis_size):
            g_c_prob = K.mean(x, axis=other_axis)  # B,W,NMAP
            g_c_prob = K.softmax(g_c_prob, axis=1)  # B,W,NMAP
            coord_pt = np.linspace(-1.0, 1.0, axis_size, dtype=np.float32) # W
            coord_pt = coord_pt.reshape([1, axis_size, 1])
            coord_pt = K.constant(coord_pt)

            g_c = K.sum(g_c_prob * coord_pt, axis=1)
            return g_c, g_c_prob

        def get_gaussian_maps(mu, shape_hw, inv_std, mode='ankush'):

            mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

            y = np.linspace(-1.0, 1.0, shape_hw[0], dtype=np.float32)
            y = y.reshape ([1, 1, shape_hw[0]])
            y = K.constant(y)
            x = np.linspace(-1.0, 1.0, shape_hw[1], dtype=np.float32)
            x = x.reshape ([1, 1, shape_hw[1]])
            x = K.constant(x)

            g_y = K.exp(-K.sqrt(1e-4 + K.abs((mu_y - y) * inv_std)))
            g_x = K.exp(-K.sqrt(1e-4 + K.abs((mu_x - x) * inv_std)))

            g_y = K.expand_dims(g_y, axis=3)
            g_x = K.expand_dims(g_x, axis=2)
            g_yx = K.batch_dot(g_y, g_x)  # [B, NMAPS, H, W]
            g_yx = K.permute_dimensions(g_yx, [0, 2, 3, 1])
            return g_yx

        #get_coord()
        xshape = heatmaps_tgt.shape.as_list()
        gauss_y, gauss_y_prob = get_coord(heatmaps_tgt, 2, xshape[1])  # B,NMAP
        gauss_x, gauss_x_prob = get_coord(heatmaps_tgt, 1, xshape[2])  # B,NMAP
        gauss_mu = K.stack([gauss_y, gauss_x], axis=2)

        gauss_xy_ = get_gaussian_maps(gauss_mu, [16, 16], 1.0 / 0.1)

        gauss_xy128 = get_gaussian_maps(gauss_mu, [128, 128], 1.0 / 0.1)

        regressor_updates = self.regressor._updates.copy()
        rec_tgt = self.regressor ( K.concatenate([feat_src, gauss_xy_]) )
        regressor_updates = xor_list (regressor_updates, self.regressor._updates )

        #G_loss = K.mean( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)(img_tgt, rec_tgt) ) + \
        G_loss = K.mean( K.square(img_tgt-rec_tgt))

        #import code
        #code.interact(local=dict(globals(), **locals()))


        self.G_opt = Adam(lr=2e-3, decay=5e-4, tf_cpu_mode=tf_cpu_mode)


        G_weights = self.detector.trainable_weights + self.feature_extractor.trainable_weights + self.regressor.trainable_weights
        self.G_train = K.function ([img_src, img_tgt],[G_loss], self.G_opt.get_updates(G_loss, G_weights)+feature_extractor_updates+detector_updates+regressor_updates )


        self.G_convert = K.function ([img_src, img_tgt],[rec_tgt, gauss_mu])

        if load_weights_locally:
            pass
        #f weights_file_root is not None:
        #   weights_file_root = Path(weights_file_root)
        #lse:
        #   weights_file_root = Path(__file__).parent
        #elf.weights_path = weights_file_root / ('FUNIT_%s.h5' % (face_type_str) )
        #f load_weights:
        #   self.model.load_weights (str(self.weights_path))



    def get_model_filename_list(self):
        return [[self.detector, 'detector.h5'],
                [self.feature_extractor,     'feature_extractor.h5'],
                [self.regressor,         'regressor.h5'],
                ]

    #def save_weights(self):
    #    self.model.save_weights (str(self.weights_path))

    def train(self, img_src, img_tgt):
        G_loss, = self.G_train ([img_src, img_tgt])
        return G_loss

    def convert(self, *args, **kwargs):
        return self.G_convert(*args, **kwargs)

    @staticmethod
    def DetectorFlow(nf):
        exec (nnlib.import_all(), locals(), globals())

        def func(x):
            x = Conv2D (nf, kernel_size=7, strides=1, padding='same', kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01)  )(x)
            x = ReLU()(x)
            x = Conv2D (nf, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D (nf*2, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*2, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D (nf*4, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*4, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D (nf*8, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*8, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D (68, kernel_size=1, strides=1, padding='valid', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            return x

        return func

    @staticmethod
    def FeatureExtractorFlow(nf):
        exec (nnlib.import_all(), locals(), globals())

        def func(x):
            x = Conv2D (nf, kernel_size=7, strides=1, padding='same', kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = ReLU()(x)
            x = Conv2D (nf, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D (nf*2, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*2, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D (nf*4, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*4, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D (nf*8, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*8, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            return x

        return func

    @staticmethod
    def RegressorFlow(nf):
        exec (nnlib.import_all(), locals(), globals())

        def func(x):

            x = Conv2D (nf*8, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*8, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            
            #x = SubpixelUpscaler()(ReLU()(BatchNormalization()(Conv2D(nf*8 * 4, kernel_size=3, strides=1, padding='same', use_bias=False)(x))))
            sh = K.int_shape(x)            
            x = Lambda ( lambda x : nnlib.tf.image.resize_images(x, (sh[1]*2,sh[2]*2)), output_shape=( sh[1]*2, sh[2]*2, sh[3] ) ) (x)

            x = Conv2D (nf*4, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*4, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            
            #x = SubpixelUpscaler()(ReLU()(BatchNormalization()(Conv2D(nf*4 * 4, kernel_size=3, strides=1, padding='same', use_bias=False)(x))))
            sh = K.int_shape(x)            
            x = Lambda ( lambda x : nnlib.tf.image.resize_images(x, (sh[1]*2,sh[2]*2)), output_shape=( sh[1]*2, sh[2]*2, sh[3] ) ) (x)
            
            x = Conv2D (nf*2, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D (nf*2, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            
            #x = SubpixelUpscaler()(ReLU()(BatchNormalization()(Conv2D(nf*2 * 4, kernel_size=3, strides=1, padding='same', use_bias=False)(x))))
            sh = K.int_shape(x)            
            x = Lambda ( lambda x : nnlib.tf.image.resize_images(x, (sh[1]*2,sh[2]*2)), output_shape=( sh[1]*2, sh[2]*2, sh[3] ) ) (x)
        
            x = Conv2D (nf, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            return Conv2D (3, kernel_size=3, strides=1, padding='same', kernel_initializer=keras.initializers.TruncatedNormal(0, 0.01))(x)

        return func