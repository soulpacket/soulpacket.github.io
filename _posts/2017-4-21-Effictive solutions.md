---
layout:     post
title:      "some effective algorithm in Python"
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

