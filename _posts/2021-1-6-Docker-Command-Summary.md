---
layout: post
title: 'Docker常见指令与原理'
categories: 'Summary'
tags:
  - [Summary, Docker]
---
本文总结了一些Docker常见的指令及其背后的原理，以及一些使用示例。

## Docker操作指令

### Docker常用指令

| 指令                                          | 功能                                              |
| --------------------------------------------- | ------------------------------------------------- |
| **镜像操作**                                  |                                                   |
| docker image ls [repo prefix]                 | 列出(部分)镜像                                    |
| docker image rm [选项] <镜像1> [<镜像2> ...]  | 删除镜像                                          |
| docker run -dit ubuntu:18.04 /bin/bash        | 运行镜像和指令<br />-d 后台运行<br />-it 分配终端 |
| **容器操作**                                  |                                                   |
| docker container ls -a                        | 列出容器(包括终止)                                |
| docker container start [container ID]         | 启动镜像                                          |
| docker container logs [container ID or NAMES] | 查看容器输出                                      |
| docker attach [container ID]                  | 进入容器                                          |
| docker exec -it [CONTAINER id] base           | 镜像执行指令                                      |
| docker container rm -f [container id]         | 删除运行中容器                                    |
| docker container prune                        | 清理所有终止容器                                  |
|                                               |                                                   |
|                                               |                                                   |
|                                               |                                                   |
|                                               |                                                   |
|                                               |                                                   |
|                                               |                                                   |
|                                               |                                                   |
|                                               |                                                   |
|                                               |                                                   |

<br>

### Dockerfile指令详解

[参考资料](https://yeasy.gitbook.io/docker_practice/image/dockerfile)

#### `COPY` 

`COPY` 指令将从构建上下文目录中 `<源路径>` 的文件/目录复制到新的一层的镜像内的 `<目标路径>` 位置。

```shell
COPY [--chown=<user>:<group>] <源路径>... <目标路径>
COPY [--chown=<user>:<group>] ["<源路径1>",... "<目标路径>"]
# e.g.
COPY package.json /usr/src/app/
```

此外，还需要注意一点，使用 `COPY` 指令，源文件的各种元数据都会保留。比如**读、写、执行权限、文件变更时间等**。这个特性对于镜像定制很有用。特别是构建相关文件都在使用 Git 进行管理的时候。

如果源路径为文件夹，**复制的时候不是直接复制该文件夹**，而是将文件夹中的内容复制到目标路径。

<br>

#### `FROM` 

`FROM` 指定 **基础镜像**，因此一个 `Dockerfile` 中 `FROM` 是必备的指令，并且必须是第一条指令。

<br>

#### `RUN`

`RUN` 指令是用来执行命令行命令的。由于命令行的强大能力，`RUN` 指令在定制镜像时是最常用的指令之一。其格式有两种：

- *shell* 格式：`RUN <命令>`，就像直接在命令行中输入的命令一样。刚才写的 Dockerfile 中的 `RUN` 指令就是这种格式。

```shell
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

- *exec* 格式：`RUN ["可执行文件", "参数1", "参数2"]`，这更像是函数调用中的格式。

<br>



<br>

## 定制镜像

### 使用 docker commit

当我们运行一个容器的时候，我们做的任何文件修改都会被记录于容器存储层里。而 Docker 提供了一个 `docker commit` 命令，可以将容器的存储层保存下来成为镜像。换句话说，就是在原有镜像的基础上，再叠加上容器的存储层，并构成新的镜像。以后我们运行这个新镜像的时候，就会拥有原有容器最后的文件变化。

`docker commit` 的语法格式为：

```shell
$ docker commit [选项] <容器ID或容器名> [<仓库名>[:<标签>]]
```

我们可以用下面的命令将容器保存为镜像：

```shell
$ docker commit \
    --author "Tao Wang <twang2218@gmail.com>" \
    --message "修改了默认网页" \
    ubuntu \
    ubuntu:v2
```

其中 `--author` 是指定修改的作者，而 `--message` 则是记录本次修改的内容。这点和 `git` 版本控制相似，不过这里这些信息可以省略留空。然后我们可以在 `docker image ls` 中看到这个新定制的镜像。

> 慎用 `docker commit`：使用 `docker commit` 命令虽然可以比较直观的帮助理解镜像分层存储的概念，但是实际环境中并不会这样使用。一般都是使用Dockerfile来定制镜像。

<br>

### 使用 Dockerfile 定制镜像

从刚才的 `docker commit` 的学习中，我们可以了解到，镜像的定制实际上就是定制每一层所添加的配置、文件。如果我们可以**把每一层修改、安装、构建、操作的命令都写入一个脚本**，用这个脚本来构建、定制镜像，那么之前提及的**无法重复的问题**、**镜像构建透明性的问题**、**体积的问题**就都会解决。这个脚本就是 Dockerfile。

Dockerfile 是一个文本文件，其内包含了一条条的 **指令(Instruction)**，**每一条指令构建一层**，因此每一条指令的内容，就是描述该层应当如何构建。因此编写命令的时候注意把多条命令用`$$`串联起来进行。

<br>

#### 镜像构建上下文 Context

我们常使用如下指令来构建镜像，如果注意，会看到 `docker build` 命令最后有一个 `.`

```shell
$ docker build -t nginx:v3 .
```

这里并不是指dockerfile所在的当前目录，而是在**指定上下文目录**。

当我们进行镜像构建的时候，并非所有定制都会通过 `RUN` 指令完成，经常会需要**将一些本地文件复制进镜像**，比如通过 `COPY` 指令、`ADD` 指令等。而 `docker build` 命令构建镜像，其实并非在本地构建，而是在服务端，也就是 Docker 引擎中构建的。那么在这种客户端/服务端的架构中，如何才能让服务端获得本地文件呢？

这就引入了上下文的概念。当构建的时候，用户会指定构建镜像上下文的路径，`docker build` 命令得知这个路径后，**会将路径下的所有内容打包，然后上传给 Docker 引擎**。这样 Docker 引擎收到这个上下文包后，展开就会获得构建镜像所需的一切文件。

如果在 `Dockerfile` 中这么写：

```dockerfile
COPY ./package.json /app/
```

这并不是要复制执行 `docker build` 命令所在的目录下的 `package.json`，也不是复制 `Dockerfile` 所在目录下的 `package.json`，而是复制 **上下文（context）** 目录下的 `package.json`。`Dockerfile` 下`COPY` 这类指令中的源文件的路径都是*相对路径*，都是基于`docker build`指令中指定的上下文目录`.`

如果有的文件不像上传到镜像中，可以使用`.dockerignore`，该文件是用于剔除不需要作为上下文传递给 Docker 引擎的。

<br>

### 使用 Git repo 进行构建

这行命令指定了构建所需的 Git repo，并且指定分支为 `master`，构建目录为 `/amd64/hello-world/`，然后 Docker 就会自己去 `git clone` 这个项目、切换到指定分支、并进入到指定目录后开始构建。

````shell
$ docker build -t hello-world https://github.com/docker-library/hello-world.git#master:amd64/hello-world
````

<br>

## 操作容器

简单的说，容器是**独立运行的一个或一组应用**，以及它们的**运行态环境**。对应的，虚拟机可以理解为模拟运行的一整套操作系统（提供了运行态环境和其他系统环境）和跑在上面的应用。

### 新建与启动

启动容器有两种方式，一种是**基于镜像新建一个容器并启动**，另外一个是将在**终止状态**（`stopped`）的容器**重新启动**。新建容器所需要的命令主要为 `docker run`。例如，下面的命令输出一个 “Hello World”，之后终止容器。

```shell
$ docker run ubuntu:18.04 /bin/echo 'Hello world'
Hello world
```

这跟在本地直接执行 `/bin/echo 'hello world'` 几乎感觉不出任何区别。

下面的命令则启动一个 bash 终端，允许用户进行交互。

```shell
$ docker run -it ubuntu:18.04 /bin/bash
```

当利用 `docker run` 来创建容器时，Docker 在后台运行的标准操作包括：

- 检查本地是否存在指定的镜像，不存在就从公有仓库下载

- 利用镜像创建并启动一个容器

- 分配一个文件系统，并在只读的镜像层外面挂载一层可读写层

- 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去

- 从地址池配置一个 ip 地址给容器

- 执行用户指定的应用程序

- 执行完毕后容器被终止

<br>

### 守护态运行

更多的时候，需要让 Docker 在后台运行而不是直接把执行命令的结果输出在当前宿主机下。此时，可以通过添加 `-d` 参数来实现。

```shell
$ docker run ubuntu:18.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"
hello world
hello world
hello world
hello world

$ docker run -d ubuntu:18.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"
77b2dc01fe0f3f1265df143181e7b9af5e05279a884f4776ee75350ea9d8017a

# 列出后台运行的容器
$ docker container ls
CONTAINER ID  IMAGE         COMMAND               CREATED        STATUS       PORTS NAMES
77b2dc01fe0f  ubuntu:18.04  /bin/sh -c 'while tr  2 minutes ago  Up 1 minute        agitated_wright

# 查看容器输出s
$ docker container logs [container ID or NAMES]
hello world
hello world
hello world
. . .
```

<br>

### 终止容器

可以使用 `docker container stop` 来终止一个运行中的容器。此外，当 Docker 容器中指定的应用终结时，容器也自动终止。例如对于只启动了一个终端的容器，用户通过 `exit` 命令或 `Ctrl+d` 来退出终端时，所创建的容器立刻终止。

终止状态的容器可以用 `docker container ls -a` 命令看到。

处于终止状态的容器，可以通过 `docker container start` 命令来重新启动。

此外，`docker container restart` 命令会将一个运行态的容器终止，然后再重新启动它。

<br>

### 进入容器

#### `attach` 命令

在使用 `-d` 参数时，容器启动后会进入后台。某些时候需要进入容器进行操作，包括使用 `docker attach` 命令或 `docker exec` 命令。

````shell
$ docker run -dit ubuntu
243c32535da7d142fb0e6df616a3c3ada0b8ab417937c853a9e1c251f499f550

$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
243c32535da7        ubuntu:latest       "/bin/bash"         18 seconds ago      Up 17 seconds                           nostalgic_hypatia

$ docker attach 243c
root@243c32535da7:/#
````

如果从这个 stdin 中 exit，会导致容器的停止。

#### `exec` 命令

`docker exec` 后边可以跟多个参数：

```shell
# exec是执行的意思，指定容器后还需有执行的指令
$ docker exec [OPTIONS] CONTAINER COMMAND

$ docker exec -i 69d1 bash
ls
# 下面为命令ls的输出，因为没有分配伪终端，因此格式是输出混乱的
bin
boot
dev
...

$ docker exec -it 69d1 bash
root@69d137adef7a:/#
```

只用 `-i` 参数时，由于没有分配伪终端，界面没有我们熟悉的 Linux 命令提示符，但命令执行结果仍然可以返回。

当 `-i` `-t` 参数一起使用时，则可以看到我们熟悉的 Linux 命令提示符。

如果从这个 stdin 中 exit，不会导致容器的停止，因为他是exec执行的一个bash。

<br>

### 删除容器

可以使用 `docker container rm` 来删除一个处于终止状态的容器

```shell
$ docker container rm [container id]

$ docker container prune	# 清理所有终止状态容器
```

如果要删除一个运行中的容器，可以添加 `-f` 参数。Docker 会发送 `SIGKILL` 信号给容器。

<br>



## Docker Compose

`Compose` 项目是 Docker 官方的开源项目，负责实现对 Docker 容器集群的快速编排。`Compose` 定位是 「定义和运行多个 Docker 容器的应用」

我们知道使用一个 `Dockerfile` 模板文件，可以让用户很方便的定义一个单独的应用容器。然而，在日常工作中，经常会碰到需要多个容器相互配合来完成某项任务的情况。例如要实现一个 Web 项目，除了 **Web 服务容器**本身，往往还需要再加上后端的**数据库服务容器**，甚至还包括负载均衡容器等。

`Compose` 恰好满足了这样的需求。它允许用户通过一个单独的 `docker-compose.yml` 模板文件（YAML 格式）来定义一组相关联的应用容器为一个项目（project）。

- 服务 (`service`)：一个应用的容器，实际上可以包括若干运行相同镜像的容器实例。
- 项目 (`project`)：由一组关联的应用容器组成的一个完整业务单元，在 `docker-compose.yml` 文件中定义。

<br>

### 安装Docker-Compose

在 Linux 上的也安装十分简单，从 [官方 GitHub Release](https://github.com/docker/compose/releases) 处直接下载编译好的二进制文件即可:

```shell
$ sudo bash -c "curl -L https://github.com/docker/compose/releases/download/1.27.4/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"

# 国内用户可以使用以下方式加快下载
$ sudo bash -c "curl -L https://download.fastgit.org/docker/compose/releases/download/1.27.4/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"

$ sudo chmod +x /usr/local/bin/docker-compose
```

<br>

### Compose 模板文件

默认的模板文件名称为 `docker-compose.yml`，格式为 YAML 格式。

````yml
version: "3"

services:
  webapp:
    image: examples/web
    ports:
      - "80:80"
    volumes:
      - "/data"
````

注意每个服务都必须通过 `image` 指令指定镜像或 `build` 指令（需要 Dockerfile）等来自动构建生成镜像。

如果使用 `build` 指令，在 `Dockerfile` 中设置的选项(例如：`CMD`, `EXPOSE`, `VOLUME`, `ENV` 等) 将会自动被获取，无需在 `docker-compose.yml` 中重复设置。

#### `build`

指定 `Dockerfile` 所在文件夹的路径（可以是绝对路径，或者相对 docker-compose.yml 文件的路径）。 `Compose` 将会利用它自动构建这个镜像，然后使用这个镜像。

````yml
version: '3'
services:

  webapp1:
    build: ./dir
    
  webapp2:
    build:
      context: ./dir
      dockerfile: Dockerfile-alternate
      args:
        buildno: 1
````

也可以使用 `context` 指令指定 `Dockerfile` 所在文件夹的路径。使用 `dockerfile` 指令指定 `Dockerfile` 文件名。使用 `arg` 指令指定构建镜像时的变量。

#### `image`

指定为镜像名称或镜像 ID。如果镜像在本地不存在，`Compose` 将会尝试拉取这个镜像。

````yml
image: ubuntu
image: orchardup/postgresql
image: a4bc65fd
````

#### `command`

覆盖容器启动后默认执行的命令。

````shell
command: echo "hello world"
````

#### `environment`

设置环境变量。你可以使用数组或字典两种格式。只给定名称的变量会自动获取运行 Compose 主机上对应变量的值，可以用来防止泄露不必要的数据。

````yml
environment:
  RACK_ENV: development
  SESSION_SECRET:

environment:
  - RACK_ENV=development
  - SESSION_SECRET
````

#### `ports`

暴露端口信息。使用宿主端口：容器端口 `(HOST:CONTAINER)` 格式，或者仅仅指定容器的端口（宿主将会随机选择端口）都可以。

```yml
ports:
 - "3000"
 - "8000:8000"
 - "49100:22"
 - "127.0.0.1:8001:8001"
```

#### `secrets`

存储敏感数据，例如 `mysql` 服务密码。

````yml
mysql:
  image: mysql
  environment:
    MYSQL_ROOT_PASSWORD_FILE: /run/secrets/db_root_password
  secrets:
    - db_root_password
    - my_other_secret
````



<br>

## Reference

- [《Docker —— 从入门到实践》](https://yeasy.gitbook.io/docker_practice/container/run)