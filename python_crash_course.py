
## VARIABLES

# Multiple assignment
x, y, name, is_cool = (1,2.6, 'John', True)

# casting
x = str(x)
y = int(y)

print(type(y),y)

## STRINGS 
name  = 'Brad'
age = 26

# contatenating
print('Hello, my name is ' + name + ' and I am ' + str(age))

# Arguments by position
print('My name is {name2} and I am {age2}'.format(name2=name, age2=age))

# F-Strings
print(f'My name is {name} and I am {age}')

# string methods
s = 'hello World'
print(s.capitalize())
print(s.swapcase())
print(s.startswith('hello'))
print(s.split()) # splits substrings into a list of strings
print(s.isalnum()) # Is all alpha numeric
print(s.isnumeric()) # true if all numeric
print(s.find('o')) # find first occurency

## LISTS. A list is a collection which ordered and changeable. Allows duplicate members

# Create a list
numbers = [1,2,3,4,5]
numbers2 = list((1,2,3,4,5))
fruits = ['Apples', 'Oranges', 'Grapes', 'Pears']

# get a value
print(fruits[1])

# get length
print(len(fruits))

# append
fruits.append('Mangos')

# remove
fruits.remove('Grapes')

# insert 
fruits.insert(2, 'Strawberries')

# pop - i.e. remove by index
fruits.pop(2)

# reverse
fruits.reverse()

# sort list
fruits.sort()
fruits.sort(reverse=True)

# change value
fruits[1] = 'Leetchi'

print(fruits)

## TUPLES. A tuple is a collection which is ordered and unchangeable. Allows duplicate members

# create a tuple
fruits = ('Apples', 'Oranges', 'Grapes')

# single value needs trailing comma
fruits2 = ('Apples',)

print(fruits2)

# delEte tuple 
del fruits

## SETS. A set is a collection which is unordered and unindexed

# Create a set
fruits_set = {'Orange', 'Pears', 'Mango'}

# Check in set
print('Oranges' in fruits_set)

# Add to set
fruits_set.add('Bananas')

# remove
fruits_set.remove('Orange')

# clear
fruits_set.clear()

# delete
del fruits_set

## DICTIONARY. A dictionary is a collection which is unordered, changeable and indexed. No duplicate members

# Create dict
person = {
	'firstname': 'John',
	'lastname': 'Doe',
	'age' : 30
}


# get value
print([person,'firstname'])

# add key/value
person['phone'] = '444 4444 222'

# get all keys or items
print(person.keys())
print(person.items())

# copy
person2 = person.copy()
person2['city'] = 'Boston'

# remove item
del(person2['age'])
person2.pop('phone')

# clear
person2.clear()

# List of dics
people = [
	{'name': 'Martha', 'age': 30},
	{'name': 'Paul', 'age': 22},
]

print(people)

## FUNCTIONS. A function is a block of code which functions when it is called.

# Create a function

def say_hello(name):
	print(f'Hello {name}')

say_hello('Pablo')

# return values
def get_sum(num1, num2):
	tot = num1 + num2
	return tot

print(get_sum(3,4))

## LANBDA FUNCTIONS. A lambda func is a small anonymous function. It can take any number of arguments, but can only have one expression.
get_sum = lambda num1, num2 : num1 + num2

print(get_sum(10,2))

## CONDITIONALS
x = 10
y = 4
 
if y>x:
	print(f'{y} is greater than {x}')
	
# Membership operator
numbers = [1,2,3,4]

if(x in numbers):
	print (x in numbers)
	
## MODULE. A module is a file containing a set of functions to include in your application. There are core python modules, modules you can install using the pip package manager, and custom modules

# import core modules
import datetime
import time
from datetime import date # import only the date object

# import pip module camelcase
from camelcase import CamelCase

today = datetime.date.today()
timestamp = time.time()

c = CamelCase()
print(c.hump('hello there world'))

print(timestamp)


## CLASSES. A class is like a blueprint for creating objects. An object has properties and methods. 

# Create a class
class User:

# ctor
	def __init__(self, name):
		self.name = name
	
	def speak(self):
		print(f'Hi I am {self.name}')
	

# Extend class
class Customer(User):
	def __init__(self, name):
		self.name = name
		self.balace = 0
		
	def set_balance(self, balance):
		self.balance = balance
		
	# OVERRIDE
	def speak(self):
		print(f'Hi I am {self.name} and my balance is {self.balance}')

	
cust1 = Customer('Pippo')
cust1.set_balance(30)
cust1.speak()
print(cust1.balance)

brad = User('Brad Jones')
brad.speak()	


# FILES. Python has functions for creating, reading, updating and deleting files

# open a file
myFile = open('myfile.txt', 'w')
print('Name:', myFile.name)

# Write to file
myFile.write('I love Python')
myFile.write('pinco pallo')
myFile.close()

# Append to file
myFile = open('myfile.txt', 'a')
myFile.write('paperino')

# Read from file
myFile = open('myfile.txt', 'r+')
text = myFile.read(100)
print(text)


## JSON. Json files are commonly used with data APIs. Here is how a json can be parsed in a Python dictionary
import json

# Sample json
userJSON = '{"first_name":"John", "last_name":"Doe"}'

# Parse json
user = json.loads(userJSON)

print(user['first_name'])