import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.gridspec as gridspec

import config

tf.enable_eager_execution

mnist=input_data.read_data_sets(os.path.join(config.ROOT_DIR,"MNIST_data"),one_hot=True)

#hyper parameters
batchSize=64
z_dim=100
x_dim=mnist.train.images.shape[1]  #the image size: 28*28
y_dim=mnist.train.labels.shape[1] # the label size(one-hot):10
h_dim=128   #hidden size of MLPs

def xavier_init(size):
    in_dim=size[0]
    xavier_stddev=1./tf.sqrt(in_dim/2.)
    return tf.random_normal(shape=size,stddev=xavier_stddev)

def sample_z(m,n):
    return np.random.uniform(-1.0,1.0,size=[m,n]).astype(np.float32)

def plot(samples):
    fig=plt.figture(figsize=(4,4))
    gs=gridspec.GridSpec(4*4)
    gs.update(wspace=0.05,hspace=0.05)
    for i, sample in enumerate(samples):
        ax =plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabel([])
        ax.set_yticklabel([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28,28),cmap='Greys_r')
    return fig

def generator_loss(logits: tf.Tensor):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.ones_like(logits)))

def discriminator_loss(real_logits:tf.Tensor,fake_logits:tf.Tensor):
    D_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,labels=tf.ones_like(real_logits)))
    D_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=tf.ones_like(fake_logits)))
    return D_loss_real+D_loss_fake
#Discriminator:
class Discriminator:
    def __init__(self, hidden_size: int =h_dim):
        self.D_W1 = tf.contrib.eager.Variable(xavier_init([x_dim + y_dim, hidden_size]))
        self.D_b1 = tf.contrib.eager.Variable(tf.zeros(shape=[hidden_size]))
        self.D_W2 = tf.contrib.eager.Variable(xavier_init(hidden_size,1))
        self.D_b2 = tf.contrib.eager.Variable(tf.zeros(shape=[1]))
        self.theta_D = [self.D_W1,self.D_W2,self.D_b1,self.D_b2]

    def forward(self,image_input,label_input):
        inputs=tf.concat(axis=1,values=[image_input,label_input])
        D_h1=tf.nn.relu(tf.matmul(inputs,self.D_W1)+self.D_b1)
        D_log_prob=tf.matmul(D_h1,self.D_W2)+self.D_b2
        D_prob=tf.nn.sigmoid(D_log_prob)
        return D_prob,D_log_prob

#Generators
class Generator:
    def __init__(self,hidden_size: int = h_dim):
        self.G_W1 = tf.contrib.eager.Variable(xavier_init([z_dim+y_dim,hidden_size]))
        self.G_b1 = tf.contrib.eager.Variable(tf.ones(shape=[hidden_size]))
        self.G_W2 = tf.contrib.eager.Variable(xavier_init([hidden_size,x_dim]))
        self.G_b2 = tf.contrib.eager.Variable(tf.ones(shape=[x_dim]))
        self.theta_G = [self.G_W1,self.G_W2,self.G_b1,self.G_b2]

    def forward(self,noise_input,label_input):
        inputs=tf.concat(axis=1,values=[noise_input,label_input])
        G_h1=tf.nn.relu(tf.matmul(inputs,self.G_W1)+self.G_b1)
        G_log_prob=tf.matmul(G_h1,self.G_W2)+self.G_b2
        G_prob=tf.nn.sigmoid(G_log_prob)
        return G_prob

def main():

    generator=Generator()
    discriminator=Discriminator()
    #setup the optimizer for two MLP networks
    D_solver=tf.train.AdadeltaOptimizer()
    G_solver=tf.train.AdadeltaOptimizer()

    out_path=os.path.join(config.OUTPUT_DIR,"conditional_gan")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #training data

    i=0
    for it in range(100000):
        if it %1000==0:
            n_sample=16
            z_sample=sample_z(n_sample,z_dim)
            y_sample=np.zeros(shape=[n_sample,y_dim])
            y_sample[:,6]=1
            samples=generator.forward(z_sample,y_sample)
            fig=plt(samples.numpy())
            file_path=os.path.join(out_path,"{}.png".format(str(i).zfill(3)))
            plt.savefig(file_path,bbox_inches='tight')
            i +=1
            plt.close(fig)

        images,labels=mnist.train.next_batch(batchSize)
        images=images.astpye(np.float32)
        labels=images.astpye(np.float32)

        z_sample=sample_z(batchSize,z_dim)

        with tf.GradientTape() as tape_d:
            g_sample=generator.forward(z_sample,labels)
            d_real,d_logit_real=discriminator.forward(images,labels)
            d_fake,d_logit_fake=discriminator.forward(g_sample,labels)
            d_loss=discriminator_loss(d_logit_real,d_logit_fake)
        # backward propagation of discriminator

        grad = tape_d.gradient(d_loss,discriminator.theta_D)
        D_solver.apply_gradients(zip(grad, discriminator.theta_D), global_step=tf.train.get_or_create_global_step())
        with tf.GradientTape() as tape_g:
            g_sample = generator.forward(z_sample, labels)
            d_fake, d_logit_fake = discriminator.forward(g_sample, labels)
            g_loss = generator_loss(d_logit_fake)

        if it % 1000 == 0:
            print("Iter: {}".format(it))
            print("D loss: {:.4}".format(d_loss))
            print("G loss: {:.4}".format(g_loss))
            print()

if __name__=='__main__':
    main()