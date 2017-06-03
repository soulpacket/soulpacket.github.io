---
layout:     post
title:      "Effective solutions in Python"
subtitle:   "record in CodeWar"
date:       2016-12-25
author:     "Mr. freedom"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - python
---

## reverse string

```python
>>> 'hello world'[::-1]
'dlrow olleh'
```

> This is extended slice syntax. It works by doing [begin:end:step] - by leaving begin and end off and specifying a step of -1, it reverses a string.

```python
''.join(reversed(s))
```

> reversed() return an object,so we need to convert type of it.

## get all indices of whatever we want in a list

```python
indices = [i for i, x in enumerate(my_list) if x == "whatever"]
```

## find the intersection of multiple sets in python

```python
u = set.intersection(set_1, set_2, set_3)
>>> list_set_all = [set_1, set_2, set_3]
>>> set.intersection(*list_set_all)
``` 

## 随机数

Description: random.randint(1, 100)生成一个随机数，我要得到一个数字和随机产生的数据相同

```python
from random import randint
class number:
	def __eq__(self, other):
		return True
guess = number()
lucky_number = randint(1, 100)
if guess == lucky_number:
	return True
```

Note: a.__eq__(b) 相当于 a == b, 这里guess.__eq__(lucky_number) 相当于 guess == lucky_number,I think it's a clever solution.

```python
from random import randint
seed(1)
guess = randint(1, 100)
seed(1)
lucky_number = randint(1, 100)
```
Note: 这里选择了相同的种子，会生成相同的随机数。

## snail

![picture](http://www.haan.lu/files/2513/8347/2456/snail.png)

```python
array = [[1,2,3],
         [8,9,4],
         [7,6,5]]
snail(array) #=> [1,2,3,4,5,6,7,8,9]
```

solution:  

```python
def snail(array):
    return list(array[0]) + snail(zip(*array[1:])[::-1]) if array else []
```

## magic class

```python
class add(int):
    def __call__(self,n):
        return add(self+n)
>>>add(1)
1
>>>add(1)(2)
3
```

* ```add``` inherit from ```int``` class, default int()=0
* add(1).__call__(2)==3 is equal with add(1)(2)


## How to generate all permutations of a iterable in python

```python
import itertools
itertools.permutations(iterable)
```