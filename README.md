# GAN
## Some  implementation  of GAN using Pytorch
## GAN.py 
  一个简单的GAN的实现，参考莫凡的教程。实现的是简单的二次曲线，用GAN来让噪音数据模拟真实二次曲线的分布，训练初期，效果如下：

![](https://github.com/cryer/GAN/raw/master/image/1.png)

随着训练的进行，后期的效果如下：

![](https://github.com/cryer/GAN/raw/master/image/2.png)

可以看到，GAN可以非常好的模拟真实数据，以至于判别器无法判断真伪。

## GAN_optim.py 
 是GAN的代码优化版本，损失函数改编成BCE损失，并且训练过程更加合理，采用交替单独训练手法。
 
 ## CGAN.py 
 是条件GAN，在GAN的基础上增加了条件，原始的GAN只是随机生成模拟的真实数据，比如模拟动物数据，
 它可以生成一个假的动物，以至于你无法分辨是不是真的，但是它无法生成你想要的动物，假如你只想生成一只狗的话。
 而CGAN就可以实现添加条件，生成你想要的条件数据。对二次曲线而言，假设上半区域的数据为1，下半区域的数据为0，这样就形成了条件，结果如下：

![](https://github.com/cryer/GAN/raw/master/image/3.png)

![](https://github.com/cryer/GAN/raw/master/image/4.png)


 ## FC_GAN.py 
 前面只是生成简单的数据，那么如果要生成图片数据呢？这个代码就是用全连接层构成生成器和判别器来生成mnist数据，效果如下：
 
 ![](https://github.com/cryer/GAN/raw/master/image/5.png)
 
 
 可以看到结果很不理想，推测是全连接层不能很好的拟合真实数据，因此采用更加合理有名的DCGAN。
 
  ## DCGAN.py 
 利用DCGAN生成mnist数据，DCGAN的介绍不多说，具体参见原论文。直接查看结果：
 
 ![](https://github.com/cryer/GAN/raw/master/image/6.png)
 
     这是训练初期
 
 ![](https://github.com/cryer/GAN/raw/master/image/7.png)
  
      这是训练后期
  
  
  `PS：注意我没有提供mnist数据集，请自行下载，放在mnist文件夹中，或者在代码加载数据集时，
  增加参数Download=True进行在线下载，但是速度很慢,另外GAN的训练速度是很慢的，因此我把训练5个小时的mnist参数给出，
  文件名为dcgan_g.pkl，需要的可以直接加载参数`
 
 下面给出50个批次的训练前中后期对比图：
 
 ![](https://github.com/cryer/GAN/raw/master/image/8.png)
 
 ![](https://github.com/cryer/GAN/raw/master/image/9.png)
 
 ![](https://github.com/cryer/GAN/raw/master/image/10.png)
 
 我也在cifar10数据集上进行了训练，同样给出一组效果图，分别为前期，中期和后期：
 
 
 ![](https://github.com/cryer/GAN/raw/master/image/11.png)
 
 ![](https://github.com/cryer/GAN/raw/master/image/12.png)
 
 ![](https://github.com/cryer/GAN/raw/master/image/13.png)
 
 
 
