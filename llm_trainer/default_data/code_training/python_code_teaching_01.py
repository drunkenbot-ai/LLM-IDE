# This file teaches Python with simple examples.

x = 5
y = 3
print(x + y)

# x stores 5.
# y stores 3.
# print shows the result.
# The program prints 8.


name = "Mina"
print("Hello", name)

# A string stores text.
# The variable name stores "Mina".
# The program prints a greeting.


numbers = [1, 2, 3, 4]
total = 0

for number in numbers:
    total = total + number

print(total)

# A list stores many values.
# A loop visits each value.
# total starts at 0.
# Each number is added.
# The final total is 10.


def add(a, b):
    return a + b


result = add(4, 6)
print(result)

# A function groups useful work.
# add takes two inputs.
# return sends back the answer.
# The answer is 10.


age = 12

if age >= 18:
    print("adult")
else:
    print("child")

# if checks a condition.
# The condition is false.
# The program prints child.


text = "cat"
print(len(text))

# len counts items.
# The word cat has three letters.


scores = {"Mina": 9, "Ravi": 8}
print(scores["Mina"])

# A dictionary connects keys to values.
# The key "Mina" has value 9.


try:
    value = int("42")
    print(value)
except ValueError:
    print("not a number")

# int changes text into a number when possible.
# "42" can become the number 42.
# try and except handle errors.
