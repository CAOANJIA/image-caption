# 论文Show, Attend and Tell的Pytorch实现  
原文地址：[Show, Attend and Tell: Neural Image Caption
Generation with Visual Attention](https://arxiv.org/pdf/1502.03044v2.pdf)    
实验进行中...

## 具体实现  

### create_input_files.py  
- 划分数据集，并将图像进行ANTIALIAS处理并resize成3x224x224，并生成word map  
	
### dataloader.py  
- 获取DataLoader  

### model.py  
- 模型总览  
	采用Encoder-Decoder架构和跨模态Attention机制  

- **Encoder**: ``VGG19``  
	1. 使用torchvision.models中在ImageNet上预训练好的模型  
	2. 去除FC层和最后一个maxpool层，即输出为14x14x512的特征图  
  
- **Decoder**: ``LSTM``  
	1. h0和c0利用encoder输出的特征图初始化  
	2. word embedding层与fc层均初始化，且不使用预训练词向量如GloVe  
	3. 采用dropout避免过拟合  
  
- **Attention**: ``scaled dot-product attention``  （与论文中不同）
	1. 利用hidden state经过fc层获得Q，feature map经过2个不同的fc层获得K和V ，Query size: [att_dim, 1], Key size: [num_pixels, att_dim], alpha size: [num_pixels, 1], Value size: [num_pixels, encoder_dim]  
	2. 缩放点积并依次通过relu、softmax获得attention score  
	3. 将V的每个pixel的所有channel都乘上该pixel对应的score，再经过residual层加上原始的encoder_out，然后对每个channel求和，得到最终的图像表示  
	4. 最后将word embedding与此向量concat，作为LSTM的输入  
  
- note:   每轮LSTMCell的迭代都需要做一次Attention，可认为是当前序列所注意的图像区域  

### train.py  
- 优化器  
	``Adam``，对于decoder我将前5个Epoch的lr设为5e-4，后5个Epoch的lr设为(5e-4)/2，在此设置之前我尝试了``Adadelta``（lr为1e-3和1e-2）Adam（lr为1e-2，1e-3和1e-4） 
- 损失函数  
	``cross-entropy`` + ``双重随机注意力正则化``（论文中的设定，鼓励模型既关注图像的每个部分又关注具体目标）
- 验证部分  
	1. 利用BLEU4分数作为模型选择的依据
	2. teacher forcing下的验证集分数为E0: 19.67;   E1: 20.85;    E2: 21.43;    E3: 21.72;    E4: 21.90;    E5: 22.32;    E6: 22.34;    **E7: 22.47**;    E8: 22.48;    E9: 22.29  
	3. 为了防止过拟合，我选择了Epoch7对应的模型
- 其他超参数  
	我将预训练轮数设为10个Epoch，batch_size设为64，LSTM的hidden_size设为768，attention向量的维度设为512，词向量维度设为256，dropout设为0.5
- 训练时间  
	在单卡NVIDIA RTX A5000下，单精度训练一个Epoch约耗时34分钟
- note：  
	1. pack_padded_sequence的使用，不能写成predictions, _ = pack_padded_sequence(predictions, decode_lens, batch_first=True)的形式，而是predictions = pack_padded_sequence(predictions, decode_lens, batch_first=True)[0]  
	2. 若不fine-tune，必须将encoder的所有参数的requires_grad设为False 
	3. 截至目前，我尝试了多种lr并尝试微调VGG，训练效果不佳，我猜测是因为使用opencv简单地resize图像至3x224x224，因此我利用PIL.Image对图像重新预处理（利用crop和ANTIALIAS），后续实验将继续进行  
### finetune.py  
- 优化器  
	encoder和decoder均为Adam，学习率均设为1e-4
- 其余参数  
	与预训练阶段相同  
- note：  
	由于未经微调，模型在测试集上BLEU的得分已经较高，而微调比较耗时且可能会遇到梯度消失等问题，我并没有继续进行这一步骤，未来工作可以尝试微调

### eval.py  
- 方法  
	BeamSearch、Autoregression  
- 指标  
	BLEU4, METEOR, ROUGE, CIDEr-D, SPICE  
- 原论文得分  
	BLEU4	|METEOR  
	:---:	|:---:  
	24.3	|23.90  
- 我的模型得分  
	1. BLEU4  
		beam_size	|BLEU4  
		:---:		|:---:  
		1		|27.08  
		2		|29.24  
		3		|**29.54**  
		4		|29.78  
  		5		|29.70  
	
## 结果  
- 在不微调Encoder的情况下模型BLEU4得分高于原论文，这可能是因为我采用了scaled-dot-product attention并对数据做了简单的预处理  
- beamsize增加时BLEU分数先增加再减小，在beamsize从1到2时分数提升较明显，2到4时的分数提升并不明显，在4时达到最高（29.78）并在5时分数下降  
- 经验证，Epoch8对应的模型BLEU4得分不如Epoch7对应的模型（我选择的）的得分，证明了我的猜测和选择是正确的（验证集上BLEU4得分提升很小，防止过拟合选择Epoch7对应的模型）

## 实验心得与经验  
1. 尝试了Adam不同的lr带来的影响，不应太大也不应太小，太大很难收敛，太小训练较慢  
2. 加载预训练模型不finetune的话应设置其所有参数的requires_grad为False  
3. 图像预处理的重要性：我尝试了简单resize图片至3x224x224，效果不好；然后尝试了使用crop和ANTIALIAS过滤器，训练效果变好，因此数据预处理十分重要  
4. 参数初始化的重要性：我尝试了word embedding层和最后的fc层均匀初始化[-0.1, 0.1]，效果不好，去掉之后效果反而更佳，说明不恰当的初始化反而会让训练变困难  

## 未来工作建议  
1. 采用预训练词向量  
2. encoder可采用Resnet-101或Resnet-152  
3. 微调encoder  

## 附录	
- 参考  
	1. [Image Caption Tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)  
	2. [Karpathy's splits of MSCOCO](https://github.com/karpathy/neuraltalk2)  
- 数据集下载  
	[MSCOCO](https://cocodataset.org/#download) 
