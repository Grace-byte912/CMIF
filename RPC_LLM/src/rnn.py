import argparse
import torch.nn as nn
from torch.distributed import rpc
import torch.multiprocessing as mp
import os
import torch
from torch.distributed.optim import DistributedOptimizer
from torch import optim
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd


# 辅助函数，生成模型参数的RRef列表，该列表用于分布式优化器对模型参数进行优化
# 在本地训练中，应用程序可以调用 Module.parameters() 来获取对所有参数tensor的引用，
# 并将其传递给本地优化器以进行后续更新。但是，相同的 API 不适用于分布式训练场景，因为某些参数位于远程计算机上。
# 因此，分布式优化器不是采用参数列表 Tensors ，而是采用 RRefs 列表
def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args, kwargs=kwargs)


# 调用本地节点
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


# Encoding layers of the RNNModel
class EmbeddingTable(nn.Module):
    r"""
    Encoding layers of the RNNModel
    """
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout).to("cuda")
        self.encoder = nn.Embedding(ntoken, ninp)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)

    def forward(self, input):
        if torch.cuda.is_available():
            input = input.cuda()
        print(f"\tEmbedding in worker {rpc.get_worker_info().name}")
        return self.drop(self.encoder(input)).cpu()


class Decoder(nn.Module):
    r"""
    Decoding layers of the RNNModel
    """
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if torch.cuda.is_available():
            self.decoder = self.decoder.cuda()
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, output):
        if torch.cuda.is_available():
            output = output.cuda()
        print(f"\tDecoder in worker {rpc.get_worker_info().name}")
        return self.decoder(self.drop(output)).cpu()


class RNNModel(nn.Module):
    r"""
    A distributed RNN model which puts embedding table and decoder parameters on
    a remote parameter server, and locally holds parameters for the LSTM module.
    The structure of the RNN model is borrowed from the word language model
    example. See https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        # setup embedding table remotely
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
        # setup LSTM locally
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        # setup decoder remotely
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        # pass input to the remote embedding table and fetch emb tensor back
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        # pass output to the remote decoder and get the decoded output back
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        print(f"\tEncoder in worker {rpc.get_worker_info().name}")
        return decoded, hidden

    def parameter_rrefs(self):
        remote_params = []
        # get RRefs of embedding table
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.rnn))
        # get RRefs of decoder
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
        return remote_params


def run_trainer():
    batch = 5
    # 词汇表大小
    ntoken = 10
    # 输入特征的维度
    ninp = 2

    # 隐藏状态的维度
    nhid = 3
    # 索引数量
    nindices = 3
    nlayers = 4
    # 创建初始隐藏状态
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            # 输入的形状为[batch, nindices]
            data = torch.LongTensor(batch, nindices) % ntoken
            # 输出的目标形状为[batch, ntoken]
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # train for 10 iterations
    for epoch in range(400):
        for data, target in get_next_batch():
            # create distributed autograd context
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])
                # run distributed optimizer
                opt.step(context_id)
                # not necessary to zero grads since they are
                # accumulated into the distributed autograd context
                # which is reset every iteration.
        print("Training epoch {}".format(epoch))


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '172.16.172.34'
    os.environ['MASTER_PORT'] = '8889'

    if rank == 0:
        print("ps init!\n")
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        print("ps init finished!\n")

    else:
        print("trainer init!\n")
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        print("trainer init finished!\n")
        run_trainer()

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parameter-Server RPC based training")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=None)

    args = parser.parse_args()
    processes = []
    p = mp.Process(target=run_worker, args=(args.rank, args.world_size))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
    # 在使用mp.spawn函数时，如果没有明确设置rank的值，它会自动为每个进程分配一个唯一的rank值。
    # mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)

