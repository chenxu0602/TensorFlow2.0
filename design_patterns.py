
MyType = type("MyType", (object,), {"a": 1})
ob = MyType()
print(type(ob))
print(ob.a)
print(isinstance(ob, object))

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance

class MyClass(Singleton):
    a = 1

one = MyClass()
two = MyClass()
two.a = 3
print(one.a)

class MyOtherClass(MyClass):
    b = 2

three = MyOtherClass()