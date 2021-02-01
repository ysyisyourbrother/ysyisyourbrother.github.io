---
layout: post
title: '论文笔记｜DRL-Cloud: Deep Reinforcement Learning-Based Resource-Provisioning and Task-Scheduling for Cloud Service Providers'
categories: '论文笔记'
tags:
  - [论文笔记, 云计算, RP-TS, 强化学习]
---

本文提出了一种利用深度强化学习来进行任务调度的算法，并基于此算法提出了DRL-Cloud框架。实验对比表明该框架极大的降低了数据中心集群的电能消耗等问题。

论文地址：https://ieeexplore.ieee.org/document/8297294

## 背景介绍

云服务平台，包括谷歌云，亚马逊云等都无法避免数据中心大量的电费开支。2020年数据中心的电量花销大约有1400亿千瓦时，这需要130亿美元的电费开销。因此要提高云服务商的收益边界，减少二氧化碳排放，必须要采取措施减少数据中心的电能花费。

数据中心的能量消耗有两个最主要的部分：

1. 服务器**低效的利用率**导致了大量能量被浪费( 对大多数服务器而言，最优的电量高效利用率在70% - 80% 之间)
2. 服务器在闲置的时候浪费了大量的能量

通过服务器的联合或负载均衡等策略可以提高整体的能量利用率，通过关闭闲置的服务器和提高正在运行的服务器的利用率。不过这里存在的挑战是：这样的策略需要有大规模的可拓展性，同时要有自适应用户请求特征变化的能力。

为了能综合解决能量的花费问题，本文提出了一种 DRL-Cloud 框架，这是第一个具有高度可拓展性和自适应能力的 **RP-TS( Resource Provisioning and/or Task Scheduling )** 框架。本文引用了通用的真实计费策略，包含 **time-of-use pricing ( TOUP )** 和 **real-time pricing ( RTP )**，除此之外还使用了 **Pay-As-You-Go** 计费协议。一旦硬件的租赁期限已到，所有的任务请求都会被拒绝。

本文有四个主要的贡献： 

1. 将DRL应用在 RP-TS 系统中
2. 半马尔可夫决策过程
3. 快速收敛和高自适应能力
4. 显著提高资源利用率和电量的消耗

<br>

## 系统模型提出

系统模型由下图1中所示。包含 workload 模型，cloud platform 模型，energy consumption 模型， 和 price 模型。

![image-20210131171202026](https://ysyisyourbrother.github.io/images/posts_img/DRL-Cloud Deep Reinforcement Learning-Based ResourceProvisioning and Task Scheduling for Cloud Service Providers/image-20210131171202026.png)

<br>

### A. User Workload Model

用户工作负荷模型由多个 jobs ( user requests ) 组成，每一个都包含多个 tasks ( with dependencies )。

1. **Job Characteristics**：用户的 jobs 由多个不相交的 **DAG ( Directed  Acyclic Graphs )** 组成。一个 DAG $G_u(N_u,W_u)$包含$N_u$个顶点和$W_u$条边。每一个顶点代表一个 task ，每一条边代表当一个 父task $_i^u\phi$ 完成后数据传送到另外一个 子task $_j^u\phi$。

   $ _n^u\phi $$ _n^u\phi $

2. **Task Characteristics**：对每个 task ${}_n^u\phi$ ，请求VM的类型标记为 ${ }_n^uK$ ，它的估计执行时间为 ${_n^uL}$ ，通过 **Nephele** 近似方法可以推理。用户规定的 task 截止时间 ${_n^uT_{ddl}}$ ，CSP 调度任务开始时间 ${_n^uT_{start}}$ 。如果任务不能使用有限的资源在给定的 ${_n^uT_{ddl}}$ 时间完成，该 task 就会被立即拒绝：$_n^uT_{start} + _n^uL<_n^uT_{ddl}$。

   CSP支持V种不同的VM类型，包括 $\{VM_1,VM_2,...\}$ ，每一个$VM_v$都和一个二元组关联 $\{R^v_{CPU},R^v_{MEM}\}$，代表每个 VM 可用的CPU和内存量。同样，每一个 task 也和一个二元组关联 $\{D^v_{CPU},D^v_{MEM}\}$ ，代表所需的CPU和内存量。如果一个 task $_n^u\phi$ 被分配给一个$VM_v$，需要满足前提条件：$R_{CPU}^v\ge {_n^uD}_{CPU}$和$R_{MEM}^v\ge {_n^uD}_{MEM}$

<br>

### B. Cloud Platform Model

如图1所示，一个CSP包含M个服务器$\{\psi_{1}, \psi_{2}, \ldots, \psi_{M}\}$，临近的服务器属于同一个服务器集群。CSP拥有 F 个服务器集群，每个集群$F_f$包含$M_f$个服务器，$\sum_{f=1}^FM_f=M$。两个服务器之间的带宽表示为$B(m,m')$，设在同一个服务器内的带宽为$B(m,m)=\infty$。在每个服务器$\psi_m$也有一个二元组$\{C_{CPU}^m,C_{MEM}^m\}$，代表服务器可用的 CPU 和 内存。

服务器$\psi_m$的状态可以表示为$\{\lambda_1^m(t),\lambda_2^m(t),...,\lambda_V^m(t)\}$，$\lambda_v^m(t)$代表 v 类型的 VM 在时间 t 运行在此服务器上的数量。

<br>

### C. Energy Consumption Model

服务器$\psi_m$的**CPU利用率**$U{r^m}(t)$计算为：
$$
U r^{m}(t)=\frac{\sum_{v=1}^{V} \lambda_{v}^{m}(t) \cdot R_{C P U}^{v}}{C_{C P U}^{m}}
$$
总的能量开销由静态电能$Pwr_{st}^m$和动态电能$Pwr_{dy}^m(t)$两部分组合。静态电能一般为常数，动态电能当CPU利用率达到最优之前是线性增长的，之后则不是，计算方式如下：
$$
\left\{\begin{array}{ll}
U r^{m}(t) \cdot \alpha_{m}, & U r^{m}(t)<U r_{O p t}^{m} \\
U r_{O p t}^{m} \cdot \alpha_{m}+\left(U r^{m}(t)-U r_{O p t}^{m}\right)^{2} \cdot \beta_{m}, & U r^{m}(t) \geq U r_{O p t}^{m}
\end{array}\right.
$$
<br>

### D. Realistic Price Model

本文考虑了一种现实的计价模型 $Price(t, Pwr_{ttl}(t))$，它由 **time-of-use-pricing ( TOUP )** 和 **real-time pricing( RTP )**组成。TOUP基于每天的时段，在高峰时段费用较高，这样可以鼓励用户在非高峰时段使用。

RTP计价则如下计算：
$$
R T P\left(P w r_{t t l}(t)\right)=\left\{\begin{array}{ll}
R T P^{l}(t), & P w r_{t t l}(t)<\theta(t) \\
R T P^{h}(t), & P w r_{t t l}(t) \geq \theta(t)
\end{array}\right.
$$
其中$\theta(t)$是一个阈值，总的电能开销计算为：
$$
\text { TotalCost }=\sum_{t=1}^{T} \operatorname{Price}\left(t, P w r_{t t l}(t)\right)
$$
<br>

## DRL-Cloud设计与实现

![image-20210131230711373](https://ysyisyourbrother.github.io/images/posts_img/DRL-Cloud Deep Reinforcement Learning-Based ResourceProvisioning and Task Scheduling for Cloud Service Providers/image-20210131230711373.png)

系统整体的设计如上图2所示。首先将tasks之间的依赖去相关性，接着解藕的tasks被送去两阶段的 RP-TS 处理器中。最后能量消耗通过 真实的 price model 进行计算。

### A. Task Decorrelation

在图2最左侧的部分可以看到，用户的Jobs收集在队列$Queue_G$中，当一个 task 的所有依赖都完成后，他就会被加入 Ready 队列$Queue_\phi$并被送入 RP-TS 处理器中。

<br>

### B. Two-Stage RP-TS Processor Based on Deep Q-Learning

**第一阶段**会判断所需资源是否足够，RP-TS处理器会为task分配一个服务器集群$F_f$，并确定task开始运行的时间。

**第二阶段**会选择一个服务器$\psi_m$来运行 task $_n^u\phi$。当这个task运行完成后，$Stage_2$会发送信号告知它的父task，并将数据传送到 job 队列中。

Q-learning-based 的两阶段 RP-TS处理器如下描述：

- **Action Space**：

  - 在$Stage_1$，DQN负责从$F$个服务器集群中选择一个服务器集群$F_f$，并确定一个开始的时间$_n^uT_{start}$。因此$Stage_1$的DQN的 action space 可以表示为：$A_{Stage_1}=\{F_1^{T_1},...,F_F^{T_T}\}$

  - 在$Stage_2$，DQN负责从服务器集群中选择一个服务器，因此 action space 为$A_{Stage_2}=\{\psi_1,...,\psi_{M_f}\}$

- **State Space**：

  - action基于当前的observation $x$决定，它是由当前的服务器observation $x_{server}$和当前的task observation $x_{task}$

    当前服务器的 $x_{server}$ 描述了可用的 $C_{CPU}^v$，$C_{MEM}^v$，而 $x_{task}$ 则由$R_{CPU}^v$和$R_{MEM}^v$组成。因此，状态state是由一个序列的 actions 和 observations 组成：$s_t=x_1,a_1,x_2,a_2,...,a_{t-1},x_t$，假设这个过程会在有限步数内完成，这会导致一个大的但有限步数的**半马尔可夫决策过程( SMDP )**，且每个序列都是一个不同的状态。

    处理器基于这些序列来学习最优的**分配策略**。处理器根据当前的state决定下一个action，采取action后会从环境中获取reward，同时改变了系统的状态。此时使用 **action-reward** 的数据来训练DQN来最大化长期的reward。

- **Reward Function**：

  RP-TS处理器的目标是通过一序列的actions，最小化长期的总能量消耗。在状态$s_t$输入 $a_t$后，系统会进入$s_{t+1}$状态，并从环境中获得一个 reward $r_t$( 它是采取了$a_t$后能量消耗的**增量**，也就是当前总能量消耗减去上一步的总能量消耗 )。

  对于$Stage_1$，reward function可以表达为：
  $$
  r_{\text {Stage}_{1}}=\operatorname{Price}\left({ }_{n}^{u} T_{\text {start }}, P w r_{t t l}^{F_{f}}\left({ }_{n}^{u} T_{\text {start }}\right)-P w r_{t t l}^{F_{f}}\left(T_{\text {start }}^{p r e}\right)\right)
  $$
  类似的，在$Stage_2$，reward function 可以表达为：
  $$
  r_{\text {Stage}_{2}}=\operatorname{Price}\left({ }_{n}^{u} T_{\text {start }}, P w r_{t t l}^{\psi_{m_{f}}}\left({ }_{n}^{u} T_{\text {start }}\right)-P w r_{t t l}^{\psi_{m} f}\left(T_{\text {start }}^{\text {pre }}\right)\right)
  $$

<br>

## Algorithm For DRL-Cloud

<img src="https://ysyisyourbrother.github.io/images/posts_img/DRL-Cloud Deep Reinforcement Learning-Based ResourceProvisioning and Task Scheduling for Cloud Service Providers/image-20210201151738350.png" alt="image-20210201151738350" style="zoom: 43%;" />

<img src="https://ysyisyourbrother.github.io/images/posts_img/DRL-Cloud Deep Reinforcement Learning-Based ResourceProvisioning and Task Scheduling for Cloud Service Providers/image-20210201151814263.png" alt="image-20210201151814263" style="zoom:45%;" />

**Experience Replay**：在算法2中，每个内部循环中存储元组 $(s_{t+1},a_t,r_t,s_t)$到 replay memory $\Delta$ 中。并随机从池中放回的选取样本进行minibatch训练。这个方法重复使用数据更新权重，提高了数据的有效利用率。随机选取可以避免数据的相关性，让训练过程更加平稳。

**Target Network**：每隔$\zeta$步，将评估网络的参数更新到目标网络上

**Exploration and Exploitation**：$\epsilon-greedy$ 算法，从大的$\epsilon$开始并逐渐递减。

<br>

## 实验结果

### A. Experiment Setup

本文选取三个baseline和DRL-Cloud进行对比：

- **The Greedy Method**：CSP选择每一个option来找到能让能量增量最小的。
- **The Round-Robin ( RR ) Method**：CSP按顺序循环的分配每一个task，如果违反了SLA，则会尝试下一个，直到不被拒绝。
- **FERPTS**

实验对比基于三个指标：能量消耗，运行时间和拒绝的task数量。本文制定了两组实验：

1. 第一组设置了小规模的3000-5000的用户请求和10个服务器集群共100-300台服务器。
2. 第二组设置了较大规模的50000-200000用户请求和10-100个服务器集群工500-5000台服务器。

实验使用了真实的用户负载追踪数据：[Google cluster-usage traces](https://github.com/google/cluster-data)。它是一个由12.5k台机器组成的集群通过29天收集而来数据集。Task dependencies 和每台 VM 具有的 resources 都是随机生成的。每个job包含的tasks数量，和tasks所需要的CPU和MEM资源数则由真实数据生成。

<br>

### B. Experiments on Small-Scale Workloads and Platforms

<img src="https://ysyisyourbrother.github.io/images/posts_img/DRL-Cloud Deep Reinforcement Learning-Based ResourceProvisioning and Task Scheduling for Cloud Service Providers/image-20210201160532964.png" alt="image-20210201160532964" style="zoom:67%;" />

<br>

### C. Experiments on Large-Scale Workloads and Platforms

<img src="https://ysyisyourbrother.github.io/images/posts_img/DRL-Cloud Deep Reinforcement Learning-Based ResourceProvisioning and Task Scheduling for Cloud Service Providers/image-20210201160946393.png" alt="image-20210201160946393" style="zoom:80%;" />