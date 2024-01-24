import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import os
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
else:
    rank = -1
    world_size = -1
print(f"RANK and WORLD_SIZE in environ : {rank}/{world_size}")
torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)