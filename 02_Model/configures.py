import torch

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE': 8,
    'SEED':42,
    'DEVICE': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    # 'DEVICE': torch.device("cpu"),

}