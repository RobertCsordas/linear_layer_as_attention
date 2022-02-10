import os
import torch.distributed
from typing import Tuple
import datetime

hostlist = None


def has_extra_work(len: int, world_size: int, rank: int) -> bool:
    rem = len % world_size
    return rank < rem


def is_work_uneven(len: int, world_size: int) -> bool:
    return len % world_size != 0


def get_work_slice(len: int, world_size: int, rank: int) -> Tuple[int, int]:
    rem = len % world_size

    real_batch_size = len // world_size
    batch_offset = real_batch_size * rank + min(rank, rem)
    real_batch_size += int(rank < rem)

    return batch_offset, real_batch_size


class SLURMEnv:
    def __init__(self) -> None:
        global hostlist

        self.rank = os.getenv("SLURM_PROCID")
        self.world_size = os.getenv("SLURM_NPROCS")
        self.hostnames = os.getenv('SLURM_JOB_NODELIST')
        self.gpu_ids = os.getenv('SLURM_STEP_GPUS')
        self.job_id = os.getenv('SLURM_JOB_ID')

        self.is_distributed = self.rank is not None and self.world_size is not None and self.hostnames is not None and\
                              self.gpu_ids is not None

        if self.is_distributed:
            if hostlist is None:
                import hostlist

            self.rank = int(self.rank)
            self.world_size = int(self.world_size)
            self.hostnames = hostlist.expand_hostlist(self.hostnames)
            self.gpu_ids = self.gpu_ids.split(",")

            self.port = 12345 + int(min(self.gpu_ids))

            self.is_distributed = self.world_size > 1

    def init_env(self):
        if not self.is_distributed:
            return

        print(f"Initializing distributed environment. World size: {self.world_size}, master: {self.hostnames[0]}, "
              f"my rank {self.rank}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_ids)
        torch.distributed.init_process_group('nccl', rank=self.rank, world_size=self.world_size,
                                             init_method=f"tcp://{self.hostnames[0]}:{self.port}",
                                             timeout=datetime.timedelta(0, 6000))

    def is_master(self):
        return (not self.is_distributed) or (self.rank == 0)

    def has_extra_work(self, work_size: int) -> bool:
        return self.is_distributed and has_extra_work(work_size, self.world_size, self.rank)

    def is_work_uneven(self, work_size: int) -> bool:
        return self.is_distributed and is_work_uneven(work_size, self.world_size)
