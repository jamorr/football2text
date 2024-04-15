from towhee import pipe
import torchvision


#Takes video file & metadata

preprocess = (
    pipe.input('vid','text')
        .map('vid', 'y', lambda x: x + 1)
        .map('text', 'y', lambda x: x + 1)
        .output('y')
)


training_configs = TrainingConfig(
     xxx='some_value_xxx',
     yyy='some_value_yyy'
)
