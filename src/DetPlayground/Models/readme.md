# Design Principles

- Every network should has its own name. And it should be registered in `model_collection` in `__init__.py`
- The name of the model is also used to find the corresponding loss calculator. 
- The output of the model should be a dictionary, containing all the information needed to calculate the loss