import os
import sys
import argparse 
import numpy as np
import vgg19
import tensorflow as tf 
import skimage.io
import skimage.transform

def parse_args():
    parser = argparse.ArgumentParser(description='convert data to feature')
    parser.add_argument('--source_path', dest='source_path', help='path of the data',
                        default='.', type=str)
    parser.add_argument('--target_path', dest='target_path', help='path to store feature map',
                        default='./feature_fc', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size',
                        default='32', type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

global args
args = parse_args()

class Data:
    def __init__(self):
        self.frame_path = os.path.join(args.source_path, 'frames')
        self.label_path = os.path.join(args.source_path, 'labels')
        self.target_path = args.target_path
        self.batch_size = args.batch_size
        
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)


    def load_data(self):
        batch_data = []
        batch_path = []
        total = 0
        num = 0
        for dirpath, dirnames, filenames in os.walk(self.frame_path):
            for f in filenames:
                total += 1
        for dirpath, dirnames, filenames in os.walk(self.frame_path):
            for f in filenames:
                num += 1
                filepath = os.path.join(dirpath, f)
                img = self.load_image(filepath)
                batch_data.append(img)
                batch_path.append([dirpath, f])
                if len(batch_path) == self.batch_size:
                    batch_data = np.asarray(batch_data)
                    batch_path = np.asarray(batch_path)
                    feature_map_batch = self.extract_feature(batch_data, batch_size=self.batch_size)
                    for i in range(self.batch_size):
                        path_ = batch_path[i][0].split('/') 
                        for l in path_:
                            if l.find('train') != -1:
                                target_dir_path = '/train/'
                                break
                            elif l.find('test') != -1:
                                target_dir_path = '/test/'
                                break
                        if not os.path.exists(self.target_path + target_dir_path):
                            os.makedirs(self.target_path + target_dir_path)
                        f_name = batch_path[i][0].replace(self.frame_path + target_dir_path, '')
                        f_name = f_name.replace('/', '_') + '_' + batch_path[i][1].replace('.png', '.npy')
                        target_dir_path = self.target_path + target_dir_path
                        target_path = os.path.join(target_dir_path, f_name)
                        print('saving feature:{0}' .format(target_path))
                        np.save(target_path, feature_map_batch[i])
                    batch_data = []
                    batch_path = []
                elif num == total:
                    batch_data = np.asarray(batch_data)
                    batch_path = np.asarray(batch_path)
                    feature_map_batch = self.extract_feature(batch_data, batch_size=len(batch_data))
                    for i in range(self.batch_size):
                        path_ = batch_path[i][0].split('/') 
                        for l in path_:
                            if l.find('train') != -1:
                                target_dir_path = '/train/'
                                break
                            elif l.find('test') != -1:
                                target_dir_path = '/test/'
                                break
                        if not os.path.exists(self.target_path + target_dir_path):
                            os.makedirs(self.target_path + target_dir_path)
                        f_name = batch_path[i][0].replace(self.frame_path + target_dir_path, '')
                        f_name = f_name.replace('/', '_') + '_' + batch_path[i][1].replace('.png', '.npy')
                        target_dir_path = self.target_path + target_dir_path
                        target_path = os.path.join(target_dir_path, f_name)
                        print('saving feature:{0}' .format(target_path))
                        np.save(target_path, feature_map_batch[i])
        
    def load_image(self, path):
        print('loading image:{0}' .format(path))
        img = skimage.io.imread(path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        resized_img = skimage.transform.resize(crop_img, (224, 224))
        return resized_img

    
    def extract_feature(self, batch, batch_size):
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    images = tf.placeholder("float", [batch_size, 224, 224, 3])
                    vgg = vgg19.Vgg19()
                    with tf.name_scope("content_vgg"):
                        vgg.build(images)
                        feature_map_batch = sess.run(vgg.fc6, feed_dict={images: batch})
        return feature_map_batch


if __name__ == "__main__":
    print args
    data = Data()
    data.load_data()
