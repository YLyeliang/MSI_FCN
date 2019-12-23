# MSI_FCN
This is the Fully convolutional network with multi-scale input proposed in paper:{to be continued.}.
Since the original code in experiments is implemented by tensorflow 1.4. It's a very old tensorflow version, 
and the original code is hard to read, since all functions are implemented in single python file. All changes to the model
should manually operated at the source code and makes it less valuable for studying.
Therefore, we decided to re-implement the code, and migrate the old version to tensorflow 2. All the parts in the experiments 
and models are implemented in different modules, so as to easily change and explore the functions.