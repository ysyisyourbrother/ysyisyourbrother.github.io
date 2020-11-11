---
layout: post
title: 'Hello Ray!  Part1: Ray Core Walkthrough'
tags:
  - [ray, RL]
---

Hello Ray!

## How to start ray?

First, we need to start ray at one server node. Use this command to start ray: 

````shell
ray start --head
````

After we enter these command, ray will tell us how to view the dashboard and then right below it'll tell you how to connect to these server on another machine. Just like these:

````shell
2020-10-31 16:28:52,274	INFO services.py:1166 -- View the Ray dashboard at http://localhost:8265

--------------------
Ray runtime started.
--------------------

Next steps
  To connect to this Ray runtime from another node, run
    ray start --address='172.16.0.11:6379' --redis-password='5241590000000000'

  Alternatively, use the following Python code:
    import ray
    ray.init(address='auto', _redis_password='5241590000000000')

  If connection fails, check your firewall settings and network configuration.

  To terminate the Ray runtime, run
    ray stop
````

By default, the dashboard only listen to localhost, testing that by entering `netstat -anp | grep 8265`, which means you can not reach dashboard from another machine directly. You can add the param below to bind the dashboard server to `0.0.0.0:8080`:

````shell
ray start --head --dashboard-host 0.0.0.0 --dashboard-port 8080
````

Next, we use another conputer to connect to the node where we start ray:

````shell
ray start --address='172.16.0.11:6379' --redis-password='5241590000000000'
````

Open dashboard to see your cluster's status:

![image-20201101135934182](https://ysyisyourbrother.github.io/images/posts_img/HelloRay/1.png)



## How to parallelize your python scripts with ray?

The basic knowledge of ray you need to know:

1. Use remote **function (task)** or **class (Actors)**: [`ray.remote`]
2. Acquire result with object IDs:  [`ray.put`, `ray.get`, `ray.wait`]

### What is remote functions(tasks)?

You can easily change your local function in python scripts by adding a decorator like `@ray.remote`:

````python
# a local python function
def regular_function():
    return 1
 
 # a remote python function
 @ray.remote
 def remote_function():
 	return 1
````

There are also some slight difference when calling this function:

````python
assert regular_function() == 1
 
object_id = remote_function.remote()
assert ray.get(object_id) == 1
````

You may not get results **immediately** when you call the function. But it will return you an `object_id` and ensures that you will get the corresponding results you want when you call `ray.get(object_id)`.

This **asynchronous feature** helps ray to achieve parallelization in python with sightly change by adding a decorator:

````python
# These happen in parallel.
for _ in range(4):
	remote_function.remote()
````



### What is object ID?

In ray, we can create remote objects, and use `object_id` to refer to them. Remote objects are stored in `Object stores` of shared memory, and each node in the cluster will hace an object store. `Object_id` can be create by:

1. Call remote functions. As just shown before.

2. Call  `ray.put()`

   ````python
   y = 1
   object_id = ray.put(1)
   
   # output:
   # object_id = ObjectID(ffffffffffffffffffff02000000008001000000)
   ````

**Remote objects are immutable**, so its value can't be altered after it is generated. And this method allow remote object copy to several `Object stores` without synchronizing copies. When data is put into the object store, **it does not get automatically broadcasted to other nodes**. Data remains local until requested by another task or actor on another node.



### How to get results?

After obtaining the `object_id`, you are promised to use it to get the corresponding result by calling this `ray.get(object_id, timeout=None)`. 

````python
y = 1
obj_id = ray.put(y)
assert ray.get(obj_id) == 1
````

In addition, you can set the `timeout` to prevent the `ray.get` function from stagnating for too long.

````python
from ray.exceptions import RayTimeoutError

@ray.remote
def long_running_function()
    time.sleep(8)

obj_id = long_running_function.remote()
try:
    ray.get(obj_id, timeout=4)
except RayTimeoutError:
    print("`get` timed out.")
````



### How to check tasks' status

`ray.wait()` will return the list of two sets of ObjectIDs. The first list contains at most `num_returns` objectIDs which have been ready for returning results. And the other list contains the remaining IDs (may be ready or unready).

````python
ready_ids, remaining_ids = ray.wait(object_ids, num_returns=1, timeout=None)

# result:
# ([ObjectID(4b32d8489203ffffffff0200000000c001000000)],
#  [ObjectID(48b870e4b925ffffffff0200000000c001000000),
#   ObjectID(19a2ca523d23ffffffff0200000000c001000000),
#   ObjectID(9ae68b5e8ad7ffffffff0200000000c001000000),
#   ObjectID(2a647e922580ffffffff0200000000c001000000)])
````



### [What is remote classes?(Actors)](https://docs.ray.io/en/latest/walkthrough.html#remote-classes-actors)

Actors extend the Ray API from functions(tasks) to classes. An actor is essentially a stateful worker. State means that we can keep variables in an instance. However, in functions, variables will expire after the functions finish.

````python
# Specify required resources for an actor.
@ray.remote(num_cpus=2, num_gpus=0.5)
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

# Create an actor from this class.
counter = Counter.remote()
````

We can interact with the actor by calling its methods with the `remote` operator. We can then call `get` on the object ref to retrieve the actual value.

````python
# Call the actor.
obj_ref = counter.increment.remote()
assert ray.get(obj_ref) == 1
````

Methods called on different actors can execute in parallel, and methods called on the same actor are executed serially in the order that thay are called. Methods on the same actor will share state with one another, as shown below.

````python
# Create ten Counter actors.
counters = [Counter.remote() for _ in range(10)]

# Increment each Counter once and get the results. These tasks all happen in
# parallel.
results = ray.get([c.increment.remote() for c in counters])
print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Increment the first Counter five times. These tasks are executed serially
# and share state.
results = ray.get([counters[0].increment.remote() for _ in range(5)])
print(results)  # prints [2, 3, 4, 5, 6]
````





## References

1. https://docs.ray.io/en/master/serialization.html





## Further Reading

- [Hello Ray!  **Part1:** Ray Core Walkthrough](https://ysyisyourbrother.github.io/Hello-Ray-Part1/)

- [Hello Ray!  **Part2:** Build A Simple RL Demo](https://ysyisyourbrother.github.io/Hello-Ray-Part2/)        

- [Hello Ray!  **Part3:** Parallelize your RL model with ray](https://ysyisyourbrother.github.io/Hello-Ray-Part3/)        