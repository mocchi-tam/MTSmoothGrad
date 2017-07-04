# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.dataset import convert

import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-t', type=int, default=1,
                        help='If negative, skip training')
    parser.add_argument('--resume', '-r', type=int, default=-1,
                        help='If positive, resume the training from snapshot')
    
    parser.add_argument('--noise', '-n', type=float, default=0.3,
                        help='SmoothGrad : sigma of Gaussian kernel')
    parser.add_argument('--sample', '-s', type=int, default=100,
                        help='SmoothGrad : number of samples')
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    flag_train = False if args.train < 0 else True
    flag_resum = False if args.resume < 0 else True
    n_epoch = args.epoch if flag_train == True else 1
    
    tsm = MTModel(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize, 
                  args.noise, args.sample)
    tsm.run()

class MTNNet(chainer.Chain):
    def __init__(self, n_out):
        super(MTNNet, self).__init__()
        with self.init_scope():
            self.cnn1 = L.Convolution2D(3, 64, ksize=3, stride=1, pad=1)
            self.cnn2 = L.Convolution2D(64, 64, ksize=3, stride=2, pad=1)
            self.lin1 = L.Linear(None, 1024)
            self.lin2 = L.Linear(None, n_out)
        
    def __call__(self, x):
        h = self.cnn1(x)
        h = F.leaky_relu(h)
        
        h = self.cnn2(h)
        h = F.leaky_relu(h)
        h = F.max_pooling_2d(h, 2, stride=2, pad=1)
        h = F.dropout(h)
        
        h = self.lin1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h)
        
        y = self.lin2(h)
        return y
    
    def predict(self, x):
        return self(x)
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = F.softmax_cross_entropy(y,t)
        self.accuracy = F.accuracy(y,t)
        return y, loss

class MTModel():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize, noise, N_sample):
        self.n_epoch = n_epoch
        self.flag_train = flag_train
        self.flag_resum = flag_resum
        self.gpu = gpu
        self.noise = noise
        self.N_sample = N_sample
        
        self.model = MTNNet(10)
        
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()
        
        if self.flag_train:
            self.optimizer = chainer.optimizers.Adam()
            self.optimizer.setup(self.model)
        
        if self.flag_resum:
            try: 
                chainer.serializers.load_npz('./net/net.model', self.model)
                chainer.serializers.load_npz('./net/net.state', self.optimizer)
                print('successfully resume model')
            except:
                print('ERROR: cannot resume model')
        
        # prepare dataset
        train, test = chainer.datasets.get_cifar10()
        
        self.N_train = len(train)
        self.N_test = 1
        
        self.train_iter = chainer.iterators.SerialIterator(train, batchsize,
                                                           repeat=True, shuffle=True)
        self.test_iter = chainer.iterators.SerialIterator(test, self.N_test,
                                                          repeat=False, shuffle=False)
        
    def run(self):
        xp = np if self.gpu < 0 else chainer.cuda.cupy
        sum_accuracy = 0
        sum_loss = 0
        
        while self.train_iter.epoch < self.n_epoch:
            # train phase
            batch = self.train_iter.next()
            if self.flag_train:
                # step by step update
                x, t = convert.concat_examples(batch, self.gpu)
                
                self.model.cleargrads()
                y, loss = self.model.loss(x, t)
                loss.backward()
                self.optimizer.update()
                
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(self.model.accuracy.data) * len(t.data)
            
            # test phase
            if self.train_iter.is_new_epoch:
                print('epoch: ', self.train_iter.epoch)
                print('train mean loss: {}, accuracy: {}'.format(
                        sum_loss / self.N_train, sum_accuracy / self.N_train))
                
                sum_accuracy = 0
                sum_loss = 0
                
                batch_test = self.test_iter.next()
                x, _ = convert.concat_examples(batch_test, self.gpu)
                
                with chainer.using_config('train', False), chainer.no_backprop_mode():
                    f = self.model.predict(x)
                t = F.argmax(f).data
                
                img_org = (x[0]*255).astype(xp.uint8)
                stdev = self.noise * (xp.max(x) - xp.min(x))
                
                x_tile = xp.tile(x, (self.N_sample,1,1,1))
                noise = xp.random.normal(0, stdev, x_tile.shape).astype(xp.float32)
                x_tile = x_tile + noise
                t = np.tile(t, self.N_sample)
                
                with chainer.using_config('train', False):
                    x_tile = chainer.Variable(x_tile)
                    y, loss = self.model.loss(x_tile, t)
                    
                    x_tile.zerograd()
                    loss.backward()
                
                total_grad = xp.sum(xp.absolute(x_tile.grad),axis=(0,1))
                grad_max = xp.max(total_grad)
                grad = ((total_grad/grad_max)*255).astype(xp.uint8)
                if self.gpu >= 0:
                    grad = chainer.cuda.to_cpu(grad)
                
                img1 = cv2.cvtColor(img_org.transpose(1,2,0), cv2.COLOR_BGR2RGB)
                img1 = cv2.resize(img1, (320,320))
                
                img2 = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
                img2 = cv2.GaussianBlur(img2,(3,3),0)
                img2 = cv2.resize(img2, (320,320))
                
                img_h = cv2.hconcat([img1, img2])
                fname = 'img_sg.png'
                cv2.imwrite(fname, img_h)
                
                self.test_iter.reset()
                sum_accuracy = 0
                sum_loss = 0
                
        try:
            chainer.serializers.save_npz('net/net.model', self.model)
            chainer.serializers.save_npz('net/net.state', self.optimizer)
            print('Successfully saved model')
        except:
            print('ERROR: saving model ignored')
        
if __name__ == '__main__':
    main()
