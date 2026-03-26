import numpy

print("Hello world")

x = 12

y = x+3

print(y)

print(f"The value of y is {y}")

def userinput():
    userinput = input("Please enter a number for x: ")
    print(f"You entered x = {userinput}")
    x = int(userinput)
    print(f"Now we will add 3 to x, so y = x + 3")
    y = x+3
    print(f"y = {y}")


userinputwanted = "yes"

if userinputwanted == "yes":
    userinput()
elif userinputwanted == "no":
    print("Okay, no input wanted")
else:
    print("Invalid input")
