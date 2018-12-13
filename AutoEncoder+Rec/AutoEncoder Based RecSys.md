# AutoEncoder Based RecSys

---

## Outline

**Vanilla/Denoising AE**

1. - [ ] [2014 - Autoencoder-based collaborative fltering](https://link.springer.com/chapter/10.1007%2F978-3-319-12643-2_35)
2. - [x] 2015-Autorec: Autoencoders meet collaborative fltering
3. - [ ] 2016-Hybrid Recommender System based on Autoencoders.
4. - [ ] 2015-Collaborative Filtering with Stacked Denoising AutoEncoders and Sparse Inputs
5. - [ ] 2015-Collaborative deep learning for recommender systems. In SIGKDD. 
6. - [ ] 2016-Collaborative denoising auto-encoders for top-n recommender
systems
7. - [ ] 2018-Aentive Contextual Denoising Autoencoder for Recommendation.
8. - [x] 2017-Trust-aware Collaborative Denoising Auto-Encoder for Top-N Recommendation.
9. - [ ] 2016-Collaborative ltering and deep learning based hybrid recommendation
for cold start problem
10. - [ ] 2017-Collaborative ltering and deep learning based recommendation
system for cold start items. 
11. - [ ] 2016-Collaborative deep ranking: a hybrid pair-wise recommendation
algorithm with implicit feedback. 



**Variational AE**

1. - [ ] 2018-A Collective Variational Autoencoder for Top-N Recommendation with Side Information
2. - [ ] 2017-Collaborative Variational Autoencoder for Recommender Systems.
3. - [ ] 2018-Variational Autoencoders for Collaborative Filtering.

**Contractive AE**

- [ ] 2017-AutoSVD++: An Ecient Hybrid Collaborative Filtering Model via Contractive Autoencoders. 

**Marginalized AE**

- [ ] 2015-Deep collaborative ltering via marginalized denoising auto-encoder. 



---
## 2015-Autorec: Autoencoders meet collaborative fltering

- code
1. https://github.com/gtshs2/Autorec
2. https://github.com/ziaoang/AutoRec
3. librec

- model

1. 思想：利用AE重建评分矩阵（填补矩阵空白，也就是为观测到的数据）
2. 参数:W V mu b
3. $\hat{y}_ui = f(V_u * f(W * (r_i)+\mu) + b)$ 

- 实现

1. rating based & RMSE 
2. 训练时只有observed数据【没观测到的数据怎么表示】 ，测试时没观测到的分数=3？

>AutoRec + NCR
1. 这个模型有没有一点像 MF+NCR?【不像 像CDAE那样加userNode才是】
如果f是identity的&b=0，那么
v = g(W * (r_i)+mu)
u = V_u
$\hat{y}_ui = u*v = \sum (u_k*v_k)$
(和MLP的区别在哪里?输入不一样)
如果f和g都不是identity的，那么
v = W * (r_i)+mu
u = V_u
$\hat{y}_ui = f(u*v+b_u)$



- 对比 MLP
>和MLP的区别:输入不一样？
MLP可不可以加userNode：好像不行？含义不明确？MLP已经有那么多层那么多bu了。搞清楚每个层的物理含义？最后几层是最终编码的特征，那么，在最后一个隐层加usernode？不太行？操作不了。


- 激活函数为啥可以用sigmoid？？？？

- 和MLP的区别在哪里?输入不一样

加一个“只要超过阈值，值就都变得差不多的激活函数”

---

##  2016-Collaborative denoising auto-encoders for top-n recommender systems（CDAE）

- code：

- 与AutoRec的区别
1. 未观测数据，AutoRec不能用于TopN推荐
2. 噪声
3. user Node

**加user node就很像加b_u**
1. 参数：$W, V, b, W', b'$
$\hat{y}_ui = f(W'_u * f(W * (r_i) + V_{u} +\mu) + b)$

>如果f和g是identity的&mu=b=0，那么
**user node是conjuctive：bu**
$v = W * (r_i) + V_{u}$ 
即$v = W * (r_i)$是$v$，用户对$K$个aspect的specific阈值是$V_{u}$
$u = W'_u$ 
$\hat{y}_ui = u*v = \sum (u_k*(v_k-b_{u_k}))$
**lexicographic rule：theta**
attentional layer：$\hat{y}_ui = attention(u)*v = \sum (\frac{exp(u_k)}{\sum exp(u_k)}*(v_k-b_{u_k}))$
**全连接->attentional layer**：$\hat{y}_ui = attention(u)*f(v) = \sum (\frac{exp(u_k)}{\sum exp(u_k)}*f(v_k-b_{u_k}))$
如果这个f可以使得没过阈值的aspect的值趋近于0

**能不能在NCF那里也加一个user node**
- model
1. general framework ：rating+ranking+不同的loss
2. 有observed：负样本采样


- CDAE v.s. DAE(没有user node)

---




DL-based-Rec + NCR
1. DL-based-Rec的提升来自于NCR
2. NCR用于DL-based-Rec

矛盾：1说明DL-based-Rec天生自带NCR思想
2说明DL-based-Rec没有这种思想，需要加进去

非线性变换：找描述能力最强的latent space，这个space可以咋样你呢？数据分布一致还是啥？
NCR还是线性变换

---

DL-based-Rec在activate fun是线性的时候，效果都不如非线性（确认下），说明效果的提升来自于非线性。即找到了更好的space
但NCR还是线性变换
所以：DL-based-Rec的提升不是来自于NCR？？？？？？？？？？？？？？？？？？？

---

NCR用于DL-based-Rec

NCR:用户有最关注的aspect
DL-based-Rec：aspect变得不明确了
- MLP:u v向量被拼接到了一起
- AutoRec/CDAE：$\hat{y}_ui = f(W'_u * f(W * (r_i) + V_{u} +\mu) + b)$
- 黑箱，看卷积和max pooling