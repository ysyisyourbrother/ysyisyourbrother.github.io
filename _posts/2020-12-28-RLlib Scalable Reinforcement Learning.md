---
layout: post
title: 'RLlib: Scalable Reinforcement Learning'
categories: '强化学习'
tags:
  - [强化学习, ray, 源码学习]
---

**RLlib( Ray v1.2.0 )**是一个基于Ray的的强化学习库，他利用了Ray的分布式特性提供了一系列可拓展的、统一的API来帮助用户构建分布式强化学习应用。RLlib天然的支持已有的机器学习框架，例如Tensorflow、PyTorch，但和这些框架又很好的隔离开来。本文将对RLlib系统进行介绍，并研读源码挖掘它的分层设计模式。

<br>

## 层次设计

下图是RLlib的分层架构：

![image-20201228151140616](https://ysyisyourbrother.github.io/images/posts_img/RLlib Scalable Reinforcement Learning/image-20201228151140616.png)

RLlib抽象层将Ray提供的API，比如`Tasks`，`Actors`等封装起来，对上层应用提供支撑。RLlib提供了一些内置的成熟的算法，比如`DQN`等，并封装在RLlib Algorithms中。用户也可以自定义自己的`policy network`和`rollout`策略来替换掉RLlib Abstractions中默认选择的方法。最顶层是应用层，用户可以将算法应用在不同的环境中，比如用来玩OpenAI Gym的游戏，进行服务的调度等。

<br>

## 核心概念和组件

RLlib中包含几个核心的概念，分别是策略(`Policies`)，采样(`Samples`)和训练(`Trainer`)。

<br>

### Policies

`Policies`也就是强化学习中的策略网络(`Policies Network`)，它用来确定`agents`和环境交互时采取的`actions`。Rollout workers通过`policy`来确定`agent`的`actions`。RLlib还支持多`agents`和多`policies`的场景：在gym环境中，只有一个`agent`和一个`policy`；在多环境下，多个`agents`和多个`env`交互，并通过共同的`policy`决策，也可以通过多个`policies`来决策。

![image-20201228152951978](https://ysyisyourbrother.github.io/images/posts_img/RLlib Scalable Reinforcement Learning/image-20201228152951978.png)

<br>

### Sample Batches

不管是运行在单个进程或者是在大的集群上，在RLlib中交换的所有数据都是通过`sample batches`的形式进行。在源码中对`sample batches`有这样的介绍：

```python
"""
Wrapper around a dictionary with string keys and array-like values.
For example, {"obs": [1, 2, 3], "reward": [0, -1, 1]} is a batch of three
    samples, each with an "obs" and "reward" attribute.
"""
```

RLlib从Rollout worker中收集一批批`rollout_fragment_length`大小的数据，并拼接成`train_batch_size`大小的数据作为`SGD`训练。

在multi-agent模式下，Sample Batch将为各个policy单独收集数据。

<br>

### Training

Policies都定义了`learn_on_batch()`方法来对policy进行训练。RLlib Trainer类协调正在运行的发布和优化策略的分布式工作流。 他们通过利用Ray并行迭代器 ( `ParallelIterator` )来实现所需的计算模式来实现此目的。 下图显示了同步采样，这是这些模式中最简单的一种：

![image-20201228154626550](https://ysyisyourbrother.github.io/images/posts_img/RLlib Scalable Reinforcement Learning/image-20201228154626550.png)

<br>

## 源码学习

官方的测试代码中对RLlib中的DQN算法demo非常简单：

```python
trainer = dqn.DQNTrainer(config = dqn.DEFAULT_CONFIG.copy(), env="CartPole-v0")
# 训练模型
for i in range(num_iterations):
    results = trainer.train()
    print(results)
trainer.stop()
```

我们就从这段简单的代码作为入口，以DQN算法为自定义算法用例，来解读RLlib库源码。

<br>

### Trainer模块

Trainer模块是RLlib的核心组件，它确定了Rollout worker的工作流程，并将他们返回的数据存储在`Buffer`中并进行训练。RLlib中Trainer模块的继承关系如下图所示：

<img src="https://ysyisyourbrother.github.io/images/posts_img/RLlib Scalable Reinforcement Learning/image-20201229004418169.png" alt="image-20201229004418169" style="zoom:50%;" />

RLlib中的Trainer继承了Ray-Tune模块的Trainable类。Tune是基于的Ray的超参数调节框架，Trainable类的`train()`方法会去调用其`step()`方法，因此继承Trainable后需要去重写`step()`方法来执行一次训练需要进行的工作。在RLlib定义的Trainer类中，默认的`step()`是去执行一次用户实现的工作流：

```python
@override(Trainer)
def step(self):
    res = next(self.train_exec_impl)
    return res
```

而用户自定义的Trainer(如这里举例的内置的**DQNTrainer**)则可以通过`build_trainer()`函数进行构造，函数会创建一个`trainer_cls`类，它继承了Trainer并配置用户的自定义内容。配置完成后还通过`with_update()`函数来更新。在源码中，DQNTrainer通过以下的代码来构造：

```python
GenericOffPolicyTrainer = build_trainer(
    name="GenericOffPolicyAlgorithm",
    default_policy=None,
    get_policy_class=get_policy_class,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    execution_plan=execution_plan)

DQNTrainer = GenericOffPolicyTrainer.with_updates(
    name="DQN", default_policy=DQNTFPolicy, default_config=DEFAULT_CONFIG)
```

可以看见DQNTrainer中自定义的配置只有几个，而核心的功能则在`DEFAULT_CONFIG`、`execution_plan`和`get_policy_class`中。内部还有一些通用的方法，比如**初始化worker_set**等。

#### DEFAULT CONFIG

这里包含了一些模型训练的配置，包括`learning rate`、`rollout_fragment_length`、`train_batch_size`和`num_workers`等一些上面提到了关键参数，都可以从这个入口进行配置。配置是通过`dict`形式构建和传递的。

#### WorkerSet

在构建`trainer_cls`的时候会根据配置初始化`num_workers`数量的`remote_worker`。通过构建`WorkerSet`类，这个类要求**至少要有一个本地的worker和0至n个remote worker**。其还包含一些关键的方法，比如同步参数`sync_weights`，添加远程worker`add_workers`等。*具体见WorkerSet模块分析*。

#### Execution Plan

这个函数传入`WorkerSet`作为参数，用来控制Rollout_worker的工作流程，并将收集数据存储到Buffer中并进行SGD训练等工作，可以说是DQNTrainer最核心部分。

工作流程主要分为三个步骤：

1. **ParallelRollouts**：Operator to collect experiences in parallel from rollout workers.

   对于workers收集到的数据，它又分为三种模式来提供数据：

   - `bulk_sync`：我们从每个worker那里收集一个batch并将它们concat在一起，返回一个大的batch。( 通过ray提供的API`for_each`来组合数据)
   - `async`：收集到的batches异步的被workers返回，没有规定顺序要求。（通过ray提供的API`gather_async()`）
   - `raw`：原始模式，用户可以通过这个自定义batches的组成方式

2. **StoreToReplayBuffer**：Callable that stores data into replay buffer actors.

3. **TrainOneStep**：Callable that improves the policy and updates workers.

   它内部又通过`get_policy`获取用户配置的策略模型，通过`do_minibatch_sgd`来执行优化来优化模型。该优化是在中心化的本地执行的，执行完成后们会通过ray的API进行**参数同步**，同步到每个remote worker中：

   ```python
   weights = ray.put(self.workers.local_worker().get_weights(
                       self.policies))
   for e in self.workers.remote_workers():
   	e.set_weights.remote(weights, _get_global_vars())
   ```

#### Policy

DQNTrainer同样自定义了自己的policy network，它提供下面这个函数来让Trainer获取策略。

```python
def get_policy_class(config: TrainerConfigDict) -> Type[Policy]:
    if config["framework"] == "torch":
        from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
        return DQNTorchPolicy
    else:
        return DQNTFPolic
```

因此我们如果需要自定义policy，只需仿照dqn_torch_policy内的内容编写自己的策略即可，*具体见policy模块分析*。



