def is_power(num):
    if num <=0:
        return False
    return (num & (num-1))==0

num=int(input("Enter the number"))
if is_power(num):
    print("The given number is apower of 2")
else:
    print("The number is not power of 2")