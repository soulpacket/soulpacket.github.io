---
layout:     post
title:      "LR for userValue"
subtitle:   "不适合人类阅读，非常水的自我笔记"
date:       2016-12-24
author:     "Mr. freedom"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - python
    - 机器学习
    - MongoDB
---

>这是用LR完成的用户价值度项目

* Logistic回归用于二分类问题，面对具体的二分类问题，比如明天是否会下雨。人们通常是估计，并没有十足的把握。因此用概率来表示再适合不过了。使用了sigmoid函数映射在[0-1]之间的特性。
* sigmoid函数<img src="http://chart.googleapis.com/chart?cht=tx&chl=%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D" style="border:none;" />
他的函数图像是![这样](http://s3.51cto.com/wyfs02/M02/59/00/wKioL1TEnTiD9dUWAAB7G34zbW0271.jpg)
* 当z>0时函数值>0.5，当z<0时函数值<0.5。
* <img src="http://chart.googleapis.com/chart?cht=tx&chl=z%3Da%2B%5Cprod_%7Bi%3D1%7D%5EN%20w_i*x_i" style="border:none;" />将函数值带入sigmoid函数，得到一个[0-1]的值，我们可以理解他为成为“1”的概率。xi时已知，求wi和常数a。
* 求wi和a时用梯度上升算法，这种算法可以理解为“爬坡”，而且是向着上升速度最快的方向“爬坡”。公式为![图](http://s3.51cto.com/wyfs02/M02/59/00/wKioL1TEnTyxHegQAAAgSFtr9U4431.jpg)
* 得到的映射值即为成为付费用户的概率。源代码[我的github](https://github.com/soulpacket/pythonwindows/blob/master/logistic-regression.py)