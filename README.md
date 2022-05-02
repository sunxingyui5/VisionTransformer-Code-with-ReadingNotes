## An image is worth 16x16 words: Transformers for image recognition at scale  
**阅读地址：** [An image is worth 16x16 words: Transformers for image recognition at scale](https://readpaper.com/paper/3094502228)  
**推荐学习视频：** [ViT论文逐段精读【论文精读】](https://www.bilibili.com/video/BV15P4y137jb?spm_id_from=333.999.0.0)  
**被引用次数：**   
**官方开源：** [vision_transformer](https://github.com/google-research/vision_transformer)  

### Vision Transformer的价值  
证明了：如果在足够多的数据上去做预训练，那么也可以不需要CNN，直接用Transformer模型也能把视觉问题解决的很好  
打破了计算机视觉（Computer Vision）和自认语言处理（Natural Language Processing）在模型上的壁垒  
实现了一些在CNN上处理不够好的例子（[Intriguing Properties of Vision Transformers](https://readpaper.com/paper/3094502228)中证明）
![examples](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/examples.jpg)
以上案列在vision transformer中处理效果优于CNN  

### 摘要  
· 一个纯的Transformer直接作用于一系列图像块的时候，它也可以在图像分类这个问题上表现很好的，尤其是在当以一个大数据集做预训练，再迁移到中小数据集时，Vision Transformer可以获得跟最好的CNN相媲美的结果    
· Transformer需要更少的训练资源（即训练起来比较便宜）  

### 简介  
比较主流的方法：先去一个大型数据集上进行预训练，在一些特定领域的小数据集上做微调（[BERT](https://readpaper.com/paper/2963341956)中提出）  
Transformer在计算用于高效性，在训练上拥有可扩展性，现在已经可以训练超过100B参数的模型了（[GPT3](https://readpaper.com/paper/3030163527)）  
随着模型和数据的增长，还未看到性能饱和的现象  
问题：图像转为序列的长度太长，导致没办法把Transformer用到视觉里来  
所以需要想办法降低序列的长度：用特征图当作Transformer的输入，降低序列长度  
>孤立自注意力：用一个局部的小窗口（类似卷积操作中的卷积核）  
>轴自注意力：$N=H\times W$（序列长度=高X宽），在$H$上做一个self-attention，在$W$上做一个self-attention  

在大规模的图像识别上，传统的残差连接网络效果还是最好的  
本文想要实现的是：一个直接应用Transformer的模型，直接作用于图片，尽量做少的修改  
Vision Transformer的做法：把图片分成很多个patch，每个patch是$16\times 16$
![patch](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/patch.jpg)
把每个个patch当作一个元素，通过一个fc层，就会得到一个liner embedding，把这些当作输入传给Transformer  
一个patch等价于一个$16\times 16$的单词（点题了）  
做有监督的训练  
ViT把计算机视觉问题当成NLP去做，中间的模型和BERT完全一样  
ViT在中型大小的数据集（Image Net）上训练，如果不加比较强的约束，ViT比同等大小的ResNet其实要弱几个点的，因为与CNN相比，Transformer少了一些归纳偏置（inductive bias：一种先验知识）  
$$ CNN的归纳偏置\left\{
\begin{matrix}
 locality：假设图片上相邻的区域会有相似的特征（即卷积核感受野中像素都是邻近的） \\
    \\
 translation equivalence：即f(g(x))=g(f(x))，无论先平移还是先卷积，效果一样 
\end{matrix}
\right.
$$   
不同于CNN，Transformer只能从数据中自己学，但是大规模预训练效果比归纳偏置要好  
Vision Transformer在有足够数据去预训练的情况下，就能在下游任务上得到很好的前移学习效果，获得跟当前最好的ResNet相似，或更好的效果  

### 图片像素序列近似（approximate）  
图⽚的像素序列太⻓，直接用有点不太现实，要想用Transformer就需要做近似approximate  
>用一个小窗口local neighborhood  
>用Sparse Transformer只对一些稀疏的点去做自注意力  
>把自注意力用到不同大小的block上，在极端情况下直接走轴  
>轴注意力：先在横轴上走自注意力，再在纵轴上做自注意力，序列长度也大大减小  

特质的自注意力机制在计算机视觉上表现不错，但需要很复杂的工程去implemented efficiently，在硬件运行上  

### Vision Transformer（ViT）  
**实现流程：** 
>①把图打成patch，再排成序列  
>②每个patch会通过线性投射层（Liner Projection of Flattened Patches）操作，得到一个特征（Patch+Position Embedding，注：图像的patch是有固定位置的）  
>③送入Transformer Encoder


![ViT](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/ViT.png)  
“\*”为特殊字符，用于分类（此方法继承自[BERT](https://readpaper.com/paper/2963341956)），模型会根据“\*”的输出做一个判断  

**计算流程：**  
①图片$X:224\times 224\times 3\overset{16\times 16 patch}{\longrightarrow}16\times 16\times 3=768$（从头到尾向量长度都是768），生成patch的数量为$N=\frac{224^2}{16^2}=196$，最终生成一个维度为$196\times 768$的矩阵  
②E：Linear Projection of Flattened Patches（全连接层），维度为$768\times 768$  
③Patch Embedding：$X \cdot E=(196 \space 768)\begin{pmatrix}768 \\768\end{pmatrix}=(196\space 768)+$位置编码$\longrightarrow (197\space 768)$
![processing](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/processing.jpg)  

**Transformer Encoder：**  
模型上的解释：  
![Encoder](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/Encoder.jpg)  

### Head Type and Class Token  
对ViT来说：怎么对图片预处理和怎么对获得的输出进行后处理，很关键  
Class Token：全局对图片理解的特征（Class Token的设计是从NLP领域借鉴来的）  
得到Token输出后接一个MLP  

### ResNet和Vision Transformer在图片处理思路上的不同  
**ResNet：** 求出一个从全局对图片特征的理解 
![ResNetprocess](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/ResNetprocess.jpg)  

**Vision Transformer：** 与原始的Transformer尽可能保持一致 
![ViTprocess](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/ViTprocess.jpg)  

### Positional Embedding  
1-d：本文中使用的，同时也是NLP中常用的  
2-d：用$(11,12,13,21,22,23)$去表示图像块（用$\frac{D}{2}$描述横坐标，$\frac{D}{2}$描述纵坐标）  
Relative Positional Embedding（相对位置编码）：如2和5，相对位置是3
>因为用的是图片块而不是像素块，所以去排列组合小块，想知道这些小块的位置信息还是比较容易的，所以用什么位置编码区别不大
>![relativepositionembedding](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/relativepositionembedding.png)

Class Token也可以用Global Average Pooling去替换  
1-d Positional Embedding也可以用2-d或者相对位置编码去替换  
Vision Transformer为与Transformer保持一致，所以使用了Class Token和1-d Positional Embedding  

### Transformer Encoder推导  
公式推导：  
![mathEncoder](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/mathEncoder.jpg)  

### Inductive vias（归纳偏置）  
Vision Transformer比CNN少很多图像特有的归纳偏置  
$$ CNN的归纳偏置\left\{
\begin{matrix}
 locality：假设图片上相邻的区域会有相似的特征（即卷积核感受野中像素都是邻近的） \\
    \\
 translation equivalence：即f(g(x))=g(f(x))，无论先平移还是先卷积，效果一样 
\end{matrix}
\right.
$$  
在Vision Transformer中，只有MLP层是局部而且平移等变性的，self-attention层是全局的  
图片2-d的信息ViT没怎么用，只有在把图片切成patch和加位置编码的时候用了，之后再也没用针对图像的归纳偏置了，故在中小数据集上训练效果不如CNN  

### Hybrid Architecture（混合模型）  
Transformer全局建模的能力比较强，CNN拥有归纳偏置不用那么多的训练数据  
**混合模型：** 前面是CNN，后面是Transformer
![HybridArchitecture](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/HybridArchitecture.jpg)  

### Fine-Tuning（微调）  
如果在微调的时候能用比较大的图像尺寸，就能得到更好的结果  
但是用预训练好的ViT模型不太好去调整图像的输入尺寸，提前预训练好的位置编码可能就没有用了，这是ViT在微调时的局限性，因为用图片的位置信息做的插值，这里的尺寸改变和抽图像块是ViT里面仅有的用了2-d信息的归纳偏置  
临时的解决方案：对位置编码做一个简单的2-d插值（torch.interplate）  

### 对比试验  
**主要对比：** ResNet，ViT和混合模型的表征学习能力  
在考虑到预训练的代价（时间）时，ViT表现的非常好，能在大多数数据集上取得最好的效果，同时训练时间更少  
在大规模数据集（JFT：图片数量303million）上预训练过的ViT
![experiment](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/experiment.png)
上述结果表明：ViT在TPUv3上训练2.5k天的效果超过了ResNet训练9k天的效果  

### ViT到底需要多少数据才能训练的比较好？  
ViT在中小数据集做预训练效果远不如ResNet，因为没用那些先验知识，但在大新数据集上ViT全面超越ResNet  
经过反复对照实验，如果想预训练ViT那至少需要包含图片14million左右的数据集
![datasets](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/datasets.jpg)  
但是实验同样发现：在比较小的数据集上，混合模型的精度非常高
![hybrid](https://github.com/sunxingyui5/VisionTransformer-Code-with-ReadingNotes/blob/main/img/hybrid.png)
