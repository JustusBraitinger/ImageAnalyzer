## How to use

This script is capable of checking a whole folder or one only one image  

It is possible to decide which attributes should be checked, just change this line:
```python
  self.checks = [
            SharpnessCheck(),
            TorchPositionCheck(),
            DarknessCheck(),
            
        ]
```

This repo also includes 3 testimage folders 
