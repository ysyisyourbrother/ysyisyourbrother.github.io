---
layout: post
title: 'Ubuntu云服务器开发环境配置文档'
categories: 'Summary'
tags:
  - [Summary, Ubuntu]
---
Ubuntu主机及云服务器常用的开发配置文档记录

[toc]

## **MySQL**安装及配置

### Step1: 安装MySQL

环境信息：

- OS：Ubuntu18.04
- MySQL: 5.7.22

在`Ubuntu`中，默认情况下，只有最新版本的`MySQL`包含在`APT`软件包存储库中,要安装它，只需更新服务器上的包索引并安装默认包`apt-get`。

```shell
#命令1
sudo apt-get update
#命令2
sudo apt-get install mysql-server
```

> apt-get是ubuntu的软件包管理工具
>
> yum是centOS的软件包管理工具

<br>

### Step2: 配置MySQL

安装完`mysql`后，可以在`shell`中输入`mysql`，系统会显示数据库`Access denied`。可见`mysql`已经安装完成，但还需要进行登陆配置。

```shell
sudo mysql_secure_installation
```

接着就会进入mysql的配置流程，只需要按照需求填写密码等信息即可。

输入`systemctl status mysql.service`可以查看`mysql`运行的情况，输入`sudo mysql -uroot -p`可以进入数据库。

<br>

### Step3: 配置远程访问

在Ubuntu下MySQL缺省是只允许本地访问的，如果你要其他机器也能够访问的话，需要进行配置；

1. 首先新建用户

   ```mysql
   create user admin;
   ```

2. 为用户授权并刷新权限

   ```mysql
   GRANT ALL PRIVILEGES ON *.* TO admin@"%" IDENTIFIED BY '123456' WITH GRANT OPTION;
   flush privileges;
   ```

   其中百分号意味着访问的地址，`localhost`就是本地访问，配置成`%`就是所有主机都可连接；

此时退出mysql并重新输入也可以成功连接进入数据库了。

```shell
mysql -uadmin -p123456
```

3. 在云服务商的网站上打开`3306`的端口的防火墙

4. 尝试远程连接数据库，如果出现错误:

   ```shell
   ERROR 2003 (HY000): Can't connect to MySQL server on '182.254.228.39' (111)
   ```

   需要修改最后检查配置`/etc/mysql/mysql.conf.d`

   - 是否有配置`skip_networking`： 这使`MySQL`只能通过本机`Socket`连接，放弃对TCP/IP的监听
   - 是否有配置`bind_address=127.0.0.1`，如果是则注释后重启`mysql`

开启、关闭重启MySQL5.7的指令如下：

```python
service mysql restart      # 重启
service mysql start		   # 启动
service mysql stop		   # 停止
```



## **Python**开发环境配置

### Step1: 安装Anaconda

官网由于是境外网站，访问慢，推荐用清华大学的开源镜像站点。使用wget直接获取安装文件：

```shell
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.1-Linux-x86_64.sh
```

下载好之后直接运行文件，可能需要先给文件赋予权限:

```shell
chmod +x Anaconda3-5.3.1-Linux-x86_64.sh
./Anaconda3-5.3.1-Linux-x86_64.sh
```

最后按照安装文件流程走即可。安装完成后，还需要在bash环境中配置anaconda

```shell
vim ～/.bashrc
export PATH=/home/ubuntu/anaconda3/bin:$PATH
source ~/.bashrc
```

<br>

### Step2: 切换源

Anaconda的python包源都是国外的源，下载速度慢，有些甚至需要梯子才能获取。

```shell
conda config --show-sources			# 查看源
```

我们可以通过换源的方法加快装包速度：

```python
vim ~/.condarc					# 编辑anaconda配置文件

# 添加源
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - defaults
show_channel_urls: true

# pip 换源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

如果遇到`conda`无法安装的包，可以使用`pip`进行安装，也同样会安装到当前的虚拟环境中。如果遇到网络波动超时的问题导致安装失败，可以进行换源或设置超时时间来解决：

```shell
pip --default-timeout=1000 install -U tensorflow-gpu								# 设置超时时间
pip install -U tensorflow-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple			# 使用清华源
```

<br>

### Step3: 构建虚拟环境

`base`环境下各种包比较多，`python`在索引包的时候会更慢，因此不建议直接在`base`环境下运行应用程序。`conda`提供了简洁的创建虚拟环境的方法：

```shell
conda create -n name python=3.6
```

使用指令切换不同虚拟环境：

```python
source activate name		# 进入虚拟环境name
source deactivate			# 推出虚拟环境
conda env list				# 列出所有虚拟环境
conda install xxx			# 进入虚拟环境后使用conda或pip会将包安装到此环境下
```

<br>

## **Golang**开发环境配置

### Step1: 下载并安装Go

在下载安装包时，请浏览[Go 官方下载页面](https://yq.aliyun.com/go/articleRenderRedirect?url=https%3A%2F%2Fgolang.org%2Fdl%2F),并且检查一下是否有新的版本可用:

```shell
wget -c https://golang.org/dl/go1.15.6.linux-amd64.tar.gz					# 官网下载
wget -c https://studygolang.com/dl/golang/go1.15.6.linux-amd64.tar.gz		# 下载速度快

# 解压文件
sudo tar xfz go1.15.6.linux-amd64.tar.gz -C /usr/local
```

Ubuntu的软件包中本身也包含Go，因此可以用`apt-get`获得（但好像并不能获得最新版本的）：

```shell
sudo apt-get install golang-go				# 安装golang
sudo apt-get --purge remove golang-go		# 卸载golang
```

<br>

### Step2: 配置环境变量

接下来需要为`shell`配置全局的环境变量，这样我们就可以直接使用`go`而不用加上它的`bin`文件夹位置

```shell
#修改~/.bashrc
sudo vim ~/.bashrc
#添加Gopath路径
export GOROOT=/usr/local/go
export PATH=$GOROOT/bin:$PATH
# 激活配置
source ~/.bashrc
# 验证是否成功
go version
```

<br>

### Step3: 开启gomod

在 `v1.11 `中加入了 Go Module 作为官方包管理形式以取代原来的GOPATH方法。不过在 `v1.11` 和 `v1.12` 的 Go 版本中 `gomod` 是不能直接使用的。可以通过 `go env` 命令返回值的 `GOMOD` 字段是否为空来判断是否已经开启了 `gomod`，如果没有开启，可以通过设置环境变量开启。

```shell
go env -w GO111MODULE=on						
go env -w GOPROXY=https://goproxy.cn,direct		# 设置代理
```

<br>

## **Vim**开发环境配置

### Step1: 安装Vim

Ubuntu18.04的云服务器一般自带相对较新版本的`vim8.0`，如果想要使用`YouCompleteMe`，则需要下载`vim8.1`以上的版本。

```shell
vim --version							# 查看vim版本
sudo apt-get update       # 更新apt
sudo apt-get remove vim-common			# 卸载当前版本vim
sudo apt-get install vim				# 安装新版vim
```

<br>

### Step2: 安装vim-plug插件管理器

`vim-plug`使用github进行托管和维护，只需要下载对应的 [plug.vim](https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim) 文件并保存到 **autoload** 目录即可完成安装。

ubuntu系统下可使用以下命令快速安装`vim-plug`。

```shell
mkdir -p ~/.vim/autoload/
cd ~/.vim/autoload/
wget https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```
如果出现raw.githubusercontent无法访问的问题，可以向`sudo vim /etc/hosts` 中添加`199.232.68.133 raw.githubusercontent.com`。

#### 1.1 vim-plug配置介绍

使用vim-plug安装vim插件的方法与另外一个著名的[vim插件管理器Vundle](https://vimjc.com/vim-plugin-manager.html)非常相似，只需要在vim配置文件 `~/.vimrc` 增加以 `call plug#begin(PLUGIN_DIRECTORY)` 开始，并以 `plug#end()` 结束的配置段即可。

下面是一个典型的vim-plug的配置实例，使用了多种vim-plug相关的配置形式:

```shell
call plug#begin('~/.vim/plugged')
Plug 'junegunn/vim-easy-align'
Plug 'https://github.com/junegunn/vim-github-dashboard.git'
Plug 'SirVer/ultisnips' | Plug 'honza/vim-snippets'
Plug 'scrooloose/nerdtree', { 'on':  'NERDTreeToggle' }
Plug 'tpope/vim-fireplace', { 'for': 'clojure' }
```

1. 在上面的`vim-plug`配置中，以 `call plug#begin('~/.vim/plugged')` 标识vim-plug配置的开始并显式指定vim插件的存放路径为 `~/.vim/plugged`；
2. `Plug 'junegunn/vim-easy-align'` 使用缩写形式指定了插件在github的地址 ([https://github.com/junegunn/vim-easy-align](https://vimjc.com/[https://github.com/junegunn/vim-easy-align))；
3. `Plug 'https://github.com/junegunn/vim-github-dashboard.git'` 则用完整的URL指定插件在github的位置；
4. `Plug 'SirVer/ultisnips' | Plug 'honza/vim-snippets'` 用 | 将两个vim插件写在同一行配置中；
5. `Plug 'scrooloose/nerdtree', { 'on': 'NERDTreeToggle' }` 使用 **按需加载**，表明只有在 `NERDTreeToggle` 命令被调用时, 对应的插件才会被加载；
6. `Plug 'tpope/vim-fireplace', { 'for': 'clojure' }` 使用 **按需加载**，表明只有编辑 *clojure* 类型的文件时该插件才会被打开；

#### 1.2 使用vim-plug安装vim插件

在Vim命令行模式下，使用命令 `:PlugInstall` 可安装vim配置文件中所有配置的vim插件；也可以使用 `PlugInstall [name ...]` 来指定安装某一个或某几个vim插件。

`:PlugStatus` 可查看vim插件的当前状态，`:PlugUpdate [name ...]` 用于安装或更新对应vim插件，而`vim-plug`本身的更新则使用命令 `:PlugUpgrade`。

安装前可能会提示没有安装`git`，此时如果输入`vim-plug`的指令将会显示该指令无效，需要先输入`sudo apt-get install git`来安装。

<br>

### Step3: 配置.vimrc文件

下面是我分享我个人的配置方法：

```shell
"显示行号
set number
" 显示标尺
set ruler
" 历史纪录
set history=1000
" 输入的命令显示出来
set showcmd
" 启动显示状态行1，总是显示状态行2
set laststatus=2
" 语法高亮显示
syntax on
set fileencodings=utf-8,gb2312,gbk,cp936,latin-1
set fileencoding=utf-8
set termencoding=utf-8
set fileformat=unix
set encoding=utf-8

" 配色方案
set background=dark
" 需要下载solarized.vim   
" 可以通过 Plug 'altercation/vim-colors-solarized'下载
" 下载后要放到.vim/colors文件夹下
colorscheme solarized
" 指定配色方案是256色
set t_Co=256

set wildmenu
" 去掉有关vi一致性模式，避免以前版本的一些bug和局限，解决backspace不能使用的问题
set nocompatible
set backspace=indent,eol,start
set backspace=2
 
" 启用自动对齐功能，把上一行的对齐格式应用到下一行
set autoindent
" 依据上面的格式，智能的选择对齐方式，对于类似C语言编写很有用处
set smartindent
 
" vim禁用自动备份
set nobackup
set nowritebackup
set noswapfile
" 用空格代替tab
set expandtab
 
" 设置显示制表符的空格字符个数,改进tab缩进值，默认为8，现改为4
set tabstop=4
" 统一缩进为4，方便在开启了et后使用退格(backspace)键，每次退格将删除X个空格
set softtabstop=4
 
" 设定自动缩进为4个字符，程序中自动缩进所使用的空白长度
set shiftwidth=4
" 设置帮助文件为中文(需要安装vimcdoc文档)
set helplang=cn
 
" 显示匹配的括号
set showmatch
" 文件缩进及tab个数
au FileType python,vim setl shiftwidth=4
au FileType python,vim setl tabstop=4
" 高亮搜索的字符串
set hlsearch
" 检测文件的类型
filetype on
filetype plugin on
filetype indent on
 
" C风格缩进
set cindent

" 去掉输入错误提示声音
set noeb
" 自动保存
set autowrite
" 突出显示当前行 
set cursorline
" Set up vertical vs block cursor for insert/normal mode
" Change cursor shape between insert and normal mode in iTerm2.app
" if $TERM_PROGRAM =~ "iTerm"
let &t_SI = "\<Esc>]50;CursorShape=1\x7" " Vertical bar in insert mode
let &t_EI = "\<Esc>]50;CursorShape=0\x7" " Block in normal mode
" endif
" Change cursor shape in screen mode
if &term =~ "screen."
    let &t_ti.="\eP\e[1 q\e\\"
    let &t_SI.="\eP\e[5 q\e\\"
    let &t_EI.="\eP\e[1 q\e\\"
    let &t_te.="\eP\e[0 q\e\\"
endif

" 设置光标样式
" 进入插入模式下的光标形状
let &t_SI.="\e[5 q"
"
" " 进入替换模式下的光标形状
let &t_SR.="\e[3 q"
"
" " 从插入模式或替换模式下退出，进入普通模式后的光标形状
let &t_EI.="\e[1 q"
"
" " 进入vim时，设置普通模式下的光标形状
autocmd VimEnter * silent !echo -ne "\e[1 q"
"
" " 离开vim后，恢复shell模式下的光标形状
autocmd VimLeave * silent !echo -ne "\e[5 q"


" 共享剪贴板
set clipboard+=unnamed
" 文件被改动时自动载入"
set autoread
" 顶部底部保持3行距离
set scrolloff=3

set mouse=a

" 退出插入模式指定类型的文件自动保存
au InsertLeave *.go,*.sh,*.py write

" ========================================
" = 一些键盘映射
" ========================================
" 一键快速编译运行
map <F5> :call CompileRunGcc()<CR>

func! CompileRunGcc()
    exec "w" 
    if  &filetype == 'cpp'
        exec '!g++ % -o %< -std=c++11'
        exec '!time ./%<'
    elseif &filetype == 'python'
        exec '!time python %'
    elseif &filetype == 'go'
        exec '!go run %' 
    endif                                                                              
endfunc

" 映射pageup和pagedown
noremap <C-e> <End>
noremap <C-a> <Home>
inoremap <C-e> <End>
inoremap <C-a> <Home>

" 映射插入模式下的方向键
inoremap <C-j> <left>
inoremap <C-k> <down>
inoremap <C-l> <right>
inoremap <C-h> <up>

nnoremap j h
nnoremap k j
nnoremap i k
nnoremap h a
nnoremap a i
vnoremap j h
vnoremap k j
vnoremap i k
vnoremap h a
vnoremap a i

" 设置vim的leader键为空格
let mapleader = "\<Space>"

" 在normal模式下回车映射
nmap <CR> o<Esc>

call plug#begin('~/.vim/plugged')
" vim-plug 插件管理器  类似vundle
" Shorthand notation; fetches https://github.com/junegunn/vim-easy-align
" 可以快速对齐的插件
Plug 'junegunn/vim-easy-align'

" 用来提供一个导航目录的侧边栏
Plug 'scrooloose/nerdtree'

" 可以使 nerdtree 的 tab 更加友好些
Plug 'jistr/vim-nerdtree-tabs'

" 自动补全括号的插件，包括小括号，中括号，以及花括号
Plug 'jiangmiao/auto-pairs'

" Vim状态栏插件，包括显示行号，列号，文件类型，文件名，以及Git状态
Plug 'vim-airline/vim-airline'

" 可以在文档中显示 git 信息
Plug 'airblade/vim-gitgutter'

" 代码自动完成，安装完插件还需要额外配置才可以使用
" Plug 'Valloric/YouCompleteMe'
" Plug 'Valloric/YouCompleteMe',{'do':'python3 install.py --go-completer'}

" jedi-vim python自动补全插件
Plug 'davidhalter/jedi-vim'

" vim-go
Plug 'fatih/vim-go', { 'do': ':GoUpdateBinaries' }

" vim快速注释
" L+c+c 注释
" L+c+u 取消注释
" L+c+L 智能注释
Plug 'scrooloose/nerdcommenter'

" vim-easymotion 插件 用于快速跳转
Plug 'easymotion/vim-easymotion'

" 配色方案
" colorscheme solarized
Plug 'altercation/vim-colors-solarized'

call plug#end()


"==============================================================================
" vim-go 插件
"==============================================================================
let g:go_version_warning = 1
let g:go_highlight_types = 1
let g:go_highlight_fields = 1
let g:go_highlight_functions = 1
let g:go_highlight_function_calls = 1 
let g:go_highlight_operators = 1
let g:go_highlight_extra_types = 1
let g:go_highlight_methods = 1
let g:go_highlight_generate_tags = 1

"==============================================================================
" jedi-vim 插件 https://github.com/davidhalter/jedi-vim/blob/master/doc/jedi-vim.txt
"==============================================================================
" 禁用jedi的自动补全功能，防止和ycm冲突
" let g:jedi#show_call_signatures = 0
" 禁用jedi的函数提示当前参数功能
" let g:jedi#completions_enabled = 0

" jedi跳转
let g:jedi#goto_command = "<leader>d"
let g:jedi#goto_assignments_command = "<leader>g"



"==============================================================================
" NERDTree 插件
"==============================================================================

" 打开和关闭NERDTree快捷键
map <F2> :NERDTreeToggle<CR>
" 聚焦到NERDTree上
map tt :NERDTreeFocus<CR>
" 显示行号
let NERDTreeShowLineNumbers=1
" 打开文件时是否显示目录
let NERDTreeAutoCenter=1
" 是否显示隐藏文件
let NERDTreeShowHidden=0
" 设置宽度
" let NERDTreeWinSize=31
" 忽略一下文件的显示
let NERDTreeIgnore=['\.pyc','\~$','\.swp']
" 打开 vim 文件及显示书签列表
let NERDTreeShowBookmarks=2
" 在终端启动vim时，共享NERDTree
let g:nerdtree_tabs_open_on_console_startup=1
" 更改i和s的映射 防止和光标移动功能冲突
let g:NERDTreeMapOpenSplit=7
let g:NERDTreeMapVOpenSplit=8


"==============================================================================
" youcompleteme 插件  https://github.com/ycm-core/YouCompleteMe
"==============================================================================
" set completeopt=menu,menuone
" 光标停留不弹出doc 手动输入\+D弹出
" let g:ycm_auto_hover = 'CursorMoved'
" nmap <leader>D <plug>(YCMHover)
" 设置多少个字符才开始提示
" let g:ycm_min_num_of_chars_for_completion = 2 
" 设置语义补全数量上限
" let g:ycm_max_num_candidates = 20

"==============================================================================
"nerdcommenter 快速注释插件 注释的时候加一个空格
"==============================================================================
let g:NERDSpaceDelims=1
```

<br>

## **Nginx**安装及配置

### Step1: 安装

我们可以直接用`apt-get`安装预编译好的二进制文件，也可以重新编译源码来得到结果。后者的好处是可以引入一些第三方的配件来让`Nginx`更加强大：

```shell
sudo apt-get update
sudo apt-get install nginx

# 安装结束后可以验证是否成功
sudo nginx -v
# 进入nginx根目录
cd /etc/nginx/
```

进入`Nginx`根目录后，需要关注的是文件`nginx.conf `和文件夹` sites-available`

### Step2: 配置

打开nginx.conf文件后可以看到一些配置，其中一些关键属性如下：

- **worker_processes:** This setting defines the number of worker processes that NGINX will use. Because NGINX is single threaded, this number should usually **be equal to the number of CPU cores**.
- **worker_connections:** This is the maximum number of simultaneous connections for each worker process and tells our worker processes how many people can simultaneously be served by NGINX. The bigger it is, the more simultaneous users the NGINX will be able to serve.
- **access_log & error_log:** These are the files that NGINX will use to log any erros and access attempts. These logs are generally reviewed for debugging and troubleshooting.

你的Nginx安装往往支持多于一个网站。因此多个网站的配置会存放在文件夹` sites-available`中。

### Step3: 常用指令

一些常用的指令如下：

```
sudo service nginx start		# 开启nginx
service nginx reload			# 重新加载nginx
service nginx status			# 查看nginx状态
```

<br>

## **Docker**安装及配置

### 使用APT安装

由于 `apt` 源使用 HTTPS 以确保软件下载过程中不被篡改。因此，我们首先需要添加使用 HTTPS 传输的软件包以及 CA 证书。

```shell
$ sudo apt-get update

$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```

为了确认所下载软件包的合法性，需要添加软件源的 `GPG` 密钥。

```shell
$ curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -


# 官方源
# $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

然后，我们需要向 `sources.list` 中添加 Docker 软件源:

```shell
$ sudo add-apt-repository \
    "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
    $(lsb_release -cs) \
    stable"


# 官方源
# $ sudo add-apt-repository \
#    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
#    $(lsb_release -cs) \
#    stable"
```

更新 apt 软件包缓存，并安装 `docker-ce`：

```shell
$ sudo apt-get update

$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

<br>

### 使用脚本安装

在测试或开发环境中 Docker 官方为了简化安装流程，提供了一套便捷的安装脚本，Ubuntu 系统上可以使用这套脚本安装，另外可以通过 `--mirror` 选项使用国内源进行安装：

```shell
# $ curl -fsSL test.docker.com -o get-docker.sh
$ curl -fsSL get.docker.com -o get-docker.sh
$ sudo sh get-docker.sh --mirror Aliyun
# $ sudo sh get-docker.sh --mirror AzureChinaCloud
```

<br>

### 建立 docker 用户组

````shell
sudo groupadd docker				# 建立docker组
sudo usermod -aG docker $USER		# 将当前用户加入 docker 组：
sudo newgrp docker					# 更新
````

<br>

### 启动docker

```shell
# 启动docker
$ sudo systemctl enable docker			
$ sudo systemctl start docker

$ systemctl restart  docker				# 重启

# 停止docker
$ sudo service docker stop				
$ sudo systemctl stop docker
```

<br>

### 下载Docker-compose

在 Linux 上的也安装十分简单，从 [官方 GitHub Release](https://github.com/docker/compose/releases) 处直接下载编译好的二进制文件即可

```shell
$ sudo curl -L https://github.com/docker/compose/releases/download/1.27.4/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose

# 国内用户可以使用以下方式加快下载
$ sudo sh -c "curl -L https://download.fastgit.org/docker/compose/releases/download/1.27.4/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"

$ sudo chmod +x /usr/local/bin/docker-compose
```

<br>

## **服务端**相关配置

### Ubuntu用户管理

#### 更改用户密码

```shell
# 默认切换到root账户下
sudo su
# user 为要切换的用户名
sudo passwd user
# 输入新旧密码，更改成功
```



<br>

### SSH远程连接配置

#### 安装ssh服务

在终端中输入指令安装ssh工具：

```shell
sudo apt update
sudo apt install openssh-server

# 安装完成后ssh工具会自动启动，此时可以查看ssh状态
sudo systemctl status ssh
```

#### 设置SSH超时时间

SSH连接如果客户端长时间没有动作，SSH连接就会被服务端自动关闭，我们可以通过设置来保持长时间连接。

```shell
sudo vim /etc/ssh/sshd_config

# 找这两个配置项，去掉注释并改为
# 	服务端每隔多少秒向客户端发送一个心跳数据
ClientAliveInterval 30
# 	客户端多少次没有相应，服务器自动断掉连接
ClientAliveCountMax 86400

# 重启ssh服务
systemctl restart sshd.service
```



<br>

## **客户端**相关配置

### 远程开发配置

#### 免密登陆

首先在本地生成ssh key对。如果服务器端需要公钥,  直接把.ssh目录下的id_rsa.pub配置即可, id_rsa为私钥一定要保密

````shell
ssh-keygen -t rsa -C "youremail@example.com"
````

接着在服务端进行配置，将公钥上传到服务端指定位置：

```shell
# 将公钥内容复制进去
vim ~/.ssh/authorized_keys		# 如果没有则会创建文件

# 重启ssh服务
systemctl restart sshd.service
```

如果出现`Connection fail. Operation timeout`等问题，可以尝试把本地的`~/.ssh/known_hosts`文件删除。

