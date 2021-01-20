---
layout: post
title: '论文笔记｜A contextual-bandit approach to personalized news article recommendation'
categories: '论文笔记'
tags:
  - [论文笔记, 推荐系统, mab, cmab]
---

本文提出了disjoint和hybrid两种LinUCB算法，并应用于雅虎首页，新闻个性化推荐场景下。该算法不仅适合新闻场景，也适合任何包含上下文的多臂老虎机问题。

## 相关背景

### 符号说明

- $A_t$是arms或actions的集合
- $x_{t,a}$: 用户$u_t$和 arm a的信息，也就是上下文信息context
- $r_{t,a_t}$: 执行arm a后新的收益。它的期望取决于user $u_t$和arm $a_t$两者
- $observation(x_{t,a_t}, a_t, r_{t,a_t})$: 使用观测数据来提升arm-selection策略
- $a_t^*$是在实验$t$中具有最大期望的payoff的arm

我们的目标是：设计一个算法A以便使total payoff期望最大化。也相当于：我们会发现一个算法，以便对应各最优的arm-selection策略的regret最小化。其中 T-trail regret $R_A(T)$定义为：
$$
R_A(T)=E[\sum_{t=1}^Tr_{a,a_t^*}]-E[\sum_{t=1}^Tr_{a,a_t}]
$$


**著名的K-armed bandit**其实是cmab的一个特例：

- arm set $At$保持不变，对于所有t都包含K个arms
- user $u_t$对所有的t都相同

因此，在每个实验中的arm set和contexts两者都是常数，对于一个bandit算法来说没啥区别，因此我们可以将该类型的bandit称为是一个**context-free bandit**。

在文章推荐场景中，我们将池子里的文章看成是arms。当一篇曝光的文章被点击，获得1的收益，否则为0。一篇文档的期望payoff就是它的点击率ctr。

<br>

### 已存在的Bandit算法

**bandit problems的基本挑战是，需要对exploration和exploitation做平衡**。算法会利用（exploits）它的过往经验来选择看起来最好的arm，但看起来最优的arm可能在实际上是次优的，因为在算法A的知识(knowledge)中是不精准的（imprecision）。**为了避免这种不希望的情况，算法A必须通过实际选择看起来次优的arms来进行explore，以便收集关于它们的更多信息**。

Exploration可能会增加**short-term regret**，因为会选到一些次优的arms。然而，获得关于arms的平均payoffs信息。因此需要重新定义算法A的arms payoffs，以减小**long-term regret**为最终目标。通常，即不会存在一个纯粹的exploring，也不会存在一个纯粹的exploiting算法，需要对两者做平衡。



<br>

## 算法

### 不相交(disjoint)线性模型的LinUCB

如果一个arm a的期望payoff在 d 维特征$x_{t,a}$上是线性的，它具有一些未知系数向量$\theta_a^*$，对于所有t：
$$
E\left[r_{t, a} \mid X_{t, a}\right]=X_{t, a}^{T} \theta_{a}^{*}
\label{exception}
$$
该模型称为disjoint的原因是：**不同arms间的参数不共享(每个arm各有一组权重，与d维特征存在加权关系得到期望payoff)**。假设：

- $D_a$是在实验 t 上的一个$m \times d$ 维的设计矩阵(design matrix)，它的行臂a对应的m个文章。
- $c_a \in R^m$是相应的响应向量(corresponding response vector)，（例如：相应的m篇文章 点击/未点击 user feedback）

因此我们可以得到优化的目标为：
$$
loss = \sum_{t=1}^{T} (c_{t,a} - D_{t,a}X_{t,a})^2
$$
我们将岭回归（ridge regression）应用到训练数据$(D_a,c_a)$上，给定了系数的一个估计（即伪逆）：
$$
\hat{\theta}_{a}=\left(D_{a}^{T} D_{a}+I_{d}\right)^{-1} D_{a}^{T} c_{a}
$$
当在 $c_{a}$ 中的元素 (components) 与 $D_{a}$ 中相应行是条件独立时，它至少具有1 - $\delta$ 的概率：
$$
\label{upper bound}
\left|x_{t, a}^{T} \hat{\theta}_{a}-E\left[r_{t, a} \mid x_{t, a}\right]\right| \leq \alpha \sqrt{x_{t, a}^{T}\left(D_{a}^{T} D_{a}+I_{d}\right)^{-1} x_{t, a}}
$$
其中 $\alpha=1+\sqrt{\ln (2 / \delta) / 2}$ 是一个常数。

也就是平均的结果和最优结果的差距不会大于不等式的右侧。换句话说，上述不等式为arm a的期望payoff给出了一个合理的紧凑的UCB，从中生成一个UCB-type arm-selection策略，在每个实验t上，选择：
$$
a_{t} \equiv \operatorname{argmax}_{a \in A_{t}}\left(x_{t, a}^{T} \hat{\theta}_{a}+\alpha \sqrt{x_{t, a}^{T} A_{a}^{-1} x_{t-a}}\right)
$$
其中：$A_{a} \equiv D_{a}^{T} D_{a}+I_{d}$

算法如下：

<img src="https://ysyisyourbrother.github.io/images/posts_img/A contextual-bandit approach to personalized news article recommendation/image-20210111144953082.png" alt="image-20210111144953082" style="zoom:50%;" />

注意，在等式$\eqref{upper bound}$中给定的αα值在一些应用中会比较大，也就是误差可能会更大。因此如果对这些参数最优化有可能会产生更高的total payoffs。不同于所有的UCB方法，LinUCB总是会选择具有最高UCB的arm。

该算法也具有一些良好的性质：

1. **它的计算复杂度对于arms的数量来说是线性的，对于特征数目最多是三次方**
2. **该算法对于一个动态的arm set来说工作良好，仍能高效运行，只要$A_t$的size不能太大**。该case在许多应用中是true的。例如，在新闻文章推荐中，编辑会从一个池子中添加/移除文章，池子的size本质上是个常数。

<br>

### Hybrid线性模型的LinUCB

在许多应用中（包含新闻推荐），除了arm-specific情况之外，所有arms都会使用共享特征。例如，在新闻文章推荐中，一个用户可能只偏爱于政治文章，因而可以提供这样的一种机制。因此，同时具有共享和非共享components的特征非常有用。我们采用如下的hybrid模型来额外添加其它的线性项到等式$\eqref{exception}$的右侧：
$$
E\left[r_{t, a} \mid x_{t, a}\right]=z_{t, a}^{T} \beta^{*}+x_{t, a}^{T} \theta_{a}^{*}
$$
其中：

- $z_{t,a}\in R^k$是当前user/article组合的特征
- $\beta^*$是一个未知的系数向量，它对所有的arms都是共享的

该模型是hybrid的，广义上系数的一些参数$\beta^*$是会被所有arms共享的，而其他参数$\theta_a^*$则不会

<img src="https://ysyisyourbrother.github.io/images/posts_img/A contextual-bandit approach to personalized news article recommendation/image-20210111171506371.png" alt="image-20210111171506371" style="zoom:50%;" />

对于hybrid模型，我们不再使用LinUCB算法作为多个不相互独立的arms的置信区间，因为他们共享特征参数。由于空间限制，我们给出了算法2的伪码（第5行和第12行会计算关于系数的redge-regression的解，第13行会计算置信区间），详细导数见完整paper。这里我们只指出了重要的事实：

1. 由于算法中使用的构建块 $\left(A_{0}, b_{0}, A_{a}, B_{a}, b_{a}\right)$ 具有固定的维度，可以进行增量更新，该算法计算十分高效。
2. 另外，与arms相关联的质量（quatities）在 $A_{t}$ 中并不存在，因而在计算量上不再相
   关。
3. 最后，我们也周期性地计算和缓存了逆 $\left(A_{0}^{-1}\right.$ 和 $A_{a}^{-1}$ ), 而非在每个实验尾部来将每个 实验的计算复杂度 $O\left(d^{2}+k^{2}\right)$ 。

<br>

## 实验

Today模块是在Yahoo! Front Page（流量最大）的最显著位置的panel。在Today Module上缺省的”Featured” tab会对高质量文章（主要新闻）的1/4进行高亮（highlight）, 而4篇文章通过一个小时级别更新的由人工编辑的文章池中进行选择。一个用户可以点击在story位置上的highlightd文章，如果她对文章感兴趣会点击进入去读取更多的详情，event被看成是一次story click。

<img src="https://ysyisyourbrother.github.io/images/posts_img/A contextual-bandit approach to personalized news article recommendation/image-20210111172338068.png" alt="image-20210111172338068" style="zoom:50%;" />

<br>

#### 数据收集

我们收集了在2009年五朋的一个随机bucket上的events。在该bucket上的用户会被随机选中，每个visiting view都有一定的概率。在该bucket中，文章会从池子中随机被选中来服务给用户。为了避免在footer位置处的曝光偏差（exposure bias），我们只关注在story位置处的F1文章的用户交互。每个**用户交互event**包含了三个部分：

-  提供给用户的从池子里随机选中的文章
- user/article特征信息
- 在story位置处用户是否对该文章有点击

在5月1号的随机bucket中有4700W的events。我们使用该天的events（称为：tuning data）来进行模型验以决定最优的参数来对比每个bandit算法。接着，我们使用调过的参数在一周的event-set（称为：evaluation data，从5月03-09号）上运行这些算法，它包含了3600w的events。

<br>

#### 特征构建

我们从原始user features开始，它们通过“support”被选中。一个feature的support指的是，用户具有该feature的比例。为了减少数据中的噪声，我们只选取了具有较高值support的features。特别的，当它的support至少为0.1时，我们才使用该feature。接着，每个user最初通过一个在1000个类别型特征(categorical components)上的原始特征向量（raw feature vector）进行表示，包含了：

- 人口属性信息(demographic information)：性别（2个分类）、年龄（离散化成10个分段）
- 地理特征（geographic features）：包含世界范围和美国的大约200个大都市；
- 行为类别(behavioral categories)：大约1000个二分类别（binary categories），它们总结了用户在Yahoo!内的消费历史。

相似的，每篇文章通过一个原始的feature vector进行表示，它以相同的方式构建了100个类别型特征。这些特征包括：

- URL类别：数十个分类，从文章资源的URL中推断得到
- editor类别：数十个主题，由人工编辑打标签总结得到

首先将类别型user/article特征编码成二分类向量，接着将每个feature vector归一化成单位长度（unit length），也会使用一个值为1的常数特征来增加每个feature vector。现在，每个article和user都可以分别表示成一个关于83个条目和1193个条目的特征向量。

为了进一步减小维度，以及捕获在这些原始特征中的非线性关系，我们会基于在2008年九月收集的随机曝光数据来执行关联分布。根据之前的降维方法[13]，我们将用户特征投影到文章类目上，接着使用相似偏好将用户聚类成分组(groups)。如下：

- 我们首先通过原始的user/article features，来使用LR来拟合一个关于点击率 (click probability) 的bilinear model, 以便 $\phi_{u}^{T} W \phi_{a}$ 来近似用户u点击文章a的概率，其中 $\phi_{u}$ 和 $\phi_{a}$ 是相应的feature vectors, W是由LR最优化得到的权重矩阵。
- 通过计算 $\psi_{u}=\phi_{u}^{T} W,$ 原始的user features接着被投影到一个induced space上。这 里，用于user u, 在 $\psi_{u}$ 的第i个元素可以被解释成：用户喜欢文章的第i个类别的度 (degree) 。 在induced的 $\psi_{u}$ space中使用K-means算法将用户聚类成5个clusters。
- 最终的user feature是一个6向量 (six-vector) ： 5个条目对应于在这5个clusters中的 成员（使用一个Gaussian kernel计算，接着归一化以便他们总和一致），第6个是一 个常数特征1.

在实验t中，每篇文章a具有一个独立的6维特征 $x_{t, a}$ (包含一个常数特征1) 。与一个user feature的外积 (outer product) 给出了6x6=36个特征，表示为 $z_{t, a} \in R^{36},$ 对应于等式(6) 的共享特征，这样 $\left(z_{t, a}, x_{t, a}\right)$ 可以被用于在hybrid线性模型中。注意，特征 $z_{t, a}$ 包含了userarticle交互信息, 而 $x_{t, a}$ 只包含了用户信息。

<br>

## 参考资料

- [d0evi1 论文学习](http://d0evi1.com/linucb/#)

