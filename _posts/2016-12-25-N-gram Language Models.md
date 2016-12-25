---
layout:     post
title:      "N-gram Language Models"
subtitle:   "n-gram、add-k smooth、Word Perplexity"
date:       2016-12-25
author:     "Mr. freedom"
header-img: "img/post-bg-js-version.jpg"
tags:
    - nlp 
---

>N-Gram（有时也称为N元模型）是自然语言处理中一个非常重要的概念，通常在NLP中，人们基于一定的语料库，可以利用N-Gram来预计或者评估一个句子是否合理

## 模型
如果我们有一个由 m 个词组成的序列（或者说一个句子），我们希望算得概率 P(w1,w2,⋯,wm) ，根据链式规则，可得 <img src="http://chart.googleapis.com/chart?cht=tx&chl=P(w_1%2Cw_2%2C...%2Cw_m)%3DP(w_1)P(w_2%7Cw_1)P(w_3%7Cw_1w_2)...P(w_m%7Cw_1...2_%7Bm-1%7D)" style="border:none;" />

这个概率显然并不好算，不妨利用马尔科夫链的假设，即当前这个词仅仅跟前面几个有限的词相关，因此也就不必追溯到最开始的那个词，这样便可以大幅缩减上诉算式的长度。

* 当 n=1, 一个一元模型（unigram model)即为 <img src="http://chart.googleapis.com/chart?cht=tx&chl=p(w_%7B1%7D%2Cw_%7B2%7D%2Cw_%7B3%7D%2C...%2Cw_%7Bm%7D)%3D%5Cprod_%7Bi%3D1%7D%5Em%20p(w_i)" style="border:none;" />
* 当 n=2, 一个二元模型（bigram model)即为<img src="http://chart.googleapis.com/chart?cht=tx&chl=p(w_%7B1%7D%2Cw_%7B2%7D%2Cw_%7B3%7D%2C...%2Cw_%7Bm%7D)%3D%5Cprod_%7Bi%3D1%7D%5Em%20p(w_i%7Cw_%7Bi-1%7D)" style="border:none;" />
* 当 n=3, 一个三元模型（trigram model)即为<img src="http://chart.googleapis.com/chart?cht=tx&chl=p(w_%7B1%7D%2Cw_%7B2%7D%2Cw_%7B3%7D%2C...%2Cw_%7Bm%7D)%3D%5Cprod_%7Bi%3D1%7D%5Em%20p(w_i%7Cw_%7Bi-2%7Dw_%7Bi-1%7D)" style="border:none;" />

接下来的思路就比较明确了，可以利用最大似然法来求出一组参数，使得训练样本的概率取得最大值。

* 对于unigram model而言，其中c(w1,..,wn) 表示 n-gram w1,..,wn 在训练语料中出现的次数，M 是语料库中的总字数（例如对于 yes no no no yes 而言，M=5）<img src="http://chart.googleapis.com/chart?cht=tx&chl=p(w_%7Bi%7D)%3D%5Cfrac%7BC(w_i)%7D%7BM%7D" style="border:none;" />
* 对于bigram model而言，<img src="http://chart.googleapis.com/chart?cht=tx&chl=p(w_%7Bi%7D%7Cw_%7Bi-1%7D)%3D%5Cfrac%7BC(w_%7Bi-1%7Dw_i)%7D%7BC(w_%7Bi-1%7D%7D" style="border:none;" />

## add-k smoothing
>有研究人员用150万词的训练语料来训练 trigram 模型，然后用同样来源的测试语料来做验证，结果发现23%的 trigram 没有在训练语料中出现过。这其实就意味着上一节我们所计算的那些概率有空为 0，这就导致了数据稀疏的可能性。
>这时的解决办法，我们称之为“平滑技术”（Smoothing）或者 “减值” （Discounting），只介绍一种平滑方式add-k。

`Add-one`是最简单、最直观的一种平滑算法。既然希望没有出现过的N-Gram的概率不再是0，那就不妨规定任何一个N-Gram在训练语料至少出现一次（即规定没有出现过的N-Gram在训练语料中出现了一次），则: `countnew(n-gram)=countold(n-gram)+1`

* 于是，对于unigram模型而言，会有<img src="http://chart.googleapis.com/chart?cht=tx&chl=p_%7Badd1%7D(w_i)%3D%5Cfrac%7BC(w_i)%2B1%7D%7BM%2B%7CV%7C%7D" style="border:none;" />

**其中，M 是训练语料中所有的N-Gram的数量（token），而 V 是所有的可能的不同的N-Gram的数量（type）。**

* 例如，对于句子`<s> the rat ate the cheese </s>`，我们可以来试着计算一下经add-one平滑后的P(ate\|rat)，即 <img src="http://chart.googleapis.com/chart?cht=tx&chl=P(ate%7Crat)%3D%5Cfrac%7BC(ratate)%2B1%7D%7BC(rat)%2B%7CV%7C%7D%3D%5Cfrac%7B2%7D%7B6%7D" style="border:none;" />

* 请注意，前面我们说过 V 是所有的可能的不同的n-Gram的数量，在这个例子中，它其实就是语料库中的词汇量，而这个词汇量是不包括 \<s> 的，但却需要包括 \</s>。 对此可以这样来理解，由于符号 \<s> 表示一个句子的开始，所以评估概率 P(\<s>|w′) 是没有意义的，因为当给定单词 w′ 的情况下来评估下一个单词可能是 \<s> 的概率是没有任何意义的，因为这种情况并不会发生。但是，同样的情况对于结束符则是有意义的。
如此一来，训练语料中未出现的n-Gram的概率不再为 0，而是一个大于 0 的较小的概率值。Add-one 平滑算法确实解决了我们的问题，但显然它也并不完美。由于训练语料中未出现n-Gram数量太多，平滑后，所有未出现的n-Gram占据了整个概率分布中的一个很大的比例。因此，在NLP中，Add-one给训练语料中没有出现过的 n-Gram 分配了太多的概率空间。此外，认为所有未出现的n-Gram概率相等是否合理其实也值得商榷。而且，对于出现在训练语料中的那些n-Gram，都增加同样的频度值，这是否欠妥，我们并不能给出一个明确的答案。
[python源码](https://github.com/soulpacket/pythonwindows/tree/master/n-gram)
