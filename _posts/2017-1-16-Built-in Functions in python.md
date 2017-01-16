---
layout:     post
title:      "Built-in Functions in python"
subtitle:   "好用的内置函数"
date:       2017-1-16
author:     "Mr. freedom"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - python
---


>2017年的第一篇博客，今年要加倍努力(:立个flag

* [abs()](#1)
* [all()](#2)
* [any()](#3)
* [ascii()](#4)
* [bin()](#5)
* [chr()](#6)
* [complex()](#7)
* [delattr()](#8)
* [divmod()](#9)
* [enumerate()](#10)
* [eval()](#11)
* [filter()](#12)

<h4 id="1">abs(x)</h4>
Return the absolute value of a number. The argument may be an integer or a floating point number. If the argument is a complex number, its magnitude is returned. 

<h4 id="2">all(iterable)</h4>
Return `True` if all elements of the iterable are true (or if the iterable is empty). Equivalent to:

```python
def all(iterable)
	for element in iterable:
		if not element:
			rerurn False
	return True
```

<h4 id="3">any(iterable)</h4>
Return True if any element of the iterable is true. If the iterable is empty, return False. Equivalent to:

```python
def any(iterable):
	for element in iterable:
		if element:
			return False
	return True
```

<h4 id="4">ascii(object)</h4>
As `repr()`, return a string containing a printable representation of an object, but escape the non-ASCII characters in the string returned by repr() using \x, \u or \U escapes. This generates a string similar to that returned by repr() in Python 2.

<h4 id="5">bin(int)</h4>
Convert an integer number to a binary string. The result is a valid Python expression. If x is not a Python int object, it has to define an __index__() method that returns an integer.


<h4 id="6">chr(int)</h4>
Return the string representing a character whose Unicode code point is the integer i. For example, chr(97) returns the string 'a', while chr(8364) returns the string '€'. This is the inverse of ord().

The valid range for the argument is from 0 through 1,114,111 (0x10FFFF in base 16). ValueError will be raised if i is outside that range.

<h4 id="7">complex([real[,imag]])</h4>
Return a complex number with the value real + imag*1j or convert a string or number to a complex number. If the first parameter is a string, it will be interpreted as a complex number and the function must be called without a second parameter. The second parameter can never be a string. Each argument may be any numeric type (including complex). If imag is omitted, it defaults to zero and the constructor serves as a numeric conversion like int and float. If both arguments are omitted, returns 0j.

>**Note**: When converting from a string, the string
must not contain whitespace around the central + or - operator. For example, complex('1+2j') is fine, but complex('1 + 2j') raises ValueError

<h4 id="8">delattr(object,name)</h4>
The arguments are an object and a string. The string must be the name of one of the object’s attributes. The function deletes the named attribute, provided the object allows it. For example, `delattr(x, 'foobar')` is equivalent to `del x.foobar`.

<h4 id="9">divmod(a,b)</h4>
Take two (non complex) numbers as arguments and return a pair of numbers consisting of their quotient and remainder when using integer division. With mixed operand types, the rules for binary arithmetic operators apply. For integers, the result is the same as`(a // b, a % b)`. For floating point numbers the result is`(q, a % b)`, where q is usually `math.floor(a / b)` but may be 1 less than that. In any case `q * b + a % b` is very close to a, if a % b is non-zero it has the same sign as b, and `0 <= abs(a % b) < abs(b)`.

<h4 id="10">enumerate(iterable,start=0)</h4>
Return an enumerate object. iterable must be a sequence, an `iterator`, or some other object which supports iteration. The `__next__()` method of the iterator returned by `enumerate()` returns a tuple containing a count (from start which defaults to 0) and the values obtained from iterating over iterable.

```python
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons,start = 1))
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

Equivalent to:

```python
def enumerate(sequence, start=0):
	n = start
	for elem in sequence:
		yield n,elem
		n += 1
```

<h4 id="11">eval(expression,globals=None,locals=None)</h4>
The arguments are a string and optional globals and locals. If provided, globals must be a dictionary. If provided, locals can be any mapping object.

The expression argument is parsed and evaluated as a Python expression (technically speaking, a condition list) using the globals and locals dictionaries as global and local namespace. If the globals dictionary is present and lacks ‘__builtins__’, the current globals are copied into globals before expression is parsed. This means that expression normally has full access to the standard builtins module and restricted environments are propagated. If the locals dictionary is omitted it defaults to the globals dictionary. If both dictionaries are omitted, the expression is executed in the environment where eval() is called. The return value is the result of the evaluated expression. Syntax errors are reported as exceptions. Example:

```python
>>>x = 1
>>>eval('x+1')
>>>2
```

<h4 id="12">filter(function, iterable)</h4>
Construct an iterator from those elements of iterable for which function returns true. iterable may be either a sequence, a container which supports iteration, or an iterator. If function is `None`, the identity function is assumed, that is, all elements of iterable that are false are removed.

Note that `filter(function, iterable)` is equivalent to the generator expression `(item for item in iterable if function(item))` if function is not `None` and `(item for item in iterable if item)` if function is None.






