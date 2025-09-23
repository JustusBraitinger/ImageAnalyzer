## How to use

This script is capable of checking a whole folder

It is possible to decide which attributes should be checked, just change this line:
```python
        self.checks = [
            WeldingCheck(),
            SharpnessCheck(),
            DarknessCheck(),
            TorchPositionCheck()
        ]
```

This repository also contains three test image folders, each of which includes manually augmented images for a specific check

