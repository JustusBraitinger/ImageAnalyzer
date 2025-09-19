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

This repository also contains three test image folders, each of which includes manually augmented images for a specific check

