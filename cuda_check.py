import torch.cuda

def check():
    gpu_list = torch.cuda.get_arch_list()
    capability = torch.cuda.get_device_capability(0)
    print(f'The gpu arch list is {gpu_list}')
    print(f'The gpu capabilites are:')
    print(capability)

if __name__ == '__main__':
    check()