# def test_func(a, b, *args, **kargs):
#     print(args)
#     print(kargs)
    
# args = {"a":1, "b":2, "c":"c"}
# args.pop("x", None)
# test_func(
#     a = args.pop("a"),
#     **args
# )

class A(object):
    def foo(self):
        print("A")
        
class B(A):
    def foo(self):
        print("B")
        
class C(A, B):
    def foo(self):
        print("C")

c = C()
c.foo()