
import numpy as np
import time
import torch
import argparse
import torch.multiprocessing as mp

from dataset import get_dataloaders, get_final_test_loader
from models import ConvTM


def process_main(tm, data_info, rank, args):
    torch.set_float32_matmul_precision('medium')
    # TODO: Set num threads to be M/N!
    torch.set_num_threads(1 + args.num_workers)
    torch.manual_seed(rank+args.seed)
    
    train_loader, test_loader = get_dataloaders(data_info, rank, args)
    test_acc = train_loop(tm, train_loader, test_loader, args.epochs)
    return test_acc
    

def train_loop(tm, train_loader, test_loader, epochs):
    max_acc = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        num_labels = 10

        start = time.perf_counter()
        for ex_idx, batch in enumerate(train_loader):
            # time.sleep(0.001)
            # continue

            x, y = batch
            x, y = x.to(tm.device, non_blocking=True), y.to(tm.device, non_blocking=True)
            # continue

            # Update from true class
            tm.ta_states[:], tm.included_literals[:] = (t.clone() for t in tm.update(x, y, 1))

            # Update from false class
            rand_offsets = torch.randint(1, num_labels, (y.size(0),), device=y.device)
            other_y = ((y + rand_offsets) % num_labels)
            tm.ta_states[:], tm.included_literals[:] = (t.clone() for t in tm.update(x, other_y, 0))

        print(f"Time taken to complete epoch and train: {time.perf_counter()-start} seconds...\n")
        start = time.perf_counter()
        test_loop(tm, test_loader, num_labels)
        print(f"Time taken to test: {time.perf_counter()-start} seconds...\n")
    return max_acc
    

def test_loop(tm, test_loader, num_classes, train=False):
    correct_count = 0
    num_examples = None
    for x, y_batch in test_loader:
        x, y_batch = x.to(tm.device, non_blocking=True), y_batch.to(tm.device, non_blocking=True)
        if num_examples is None:
            num_examples = y_batch.shape[0]
        sums = []
        class_batches = [y_batch] + [(y_batch + i) % num_classes for i in range(1, num_classes)]
        for n, batch in enumerate(class_batches):
            class_sum = tm.predict(x, batch).clone()
            sums.append(class_sum)

        preds = torch.stack(sums).argmax(dim=0)
        correct_count += (preds == 0).sum()
    if train:
        print(f"Train Accuracy: {round((correct_count.item() / (len(test_loader) * num_examples)) * 100, 4)}%")
    else:
        test_acc = round((correct_count.item() / (len(test_loader) * num_examples)) * 100, 4)
        print(f"Test Accuracy: {test_acc}%")
        return test_acc


def main(args):
    train_size = len(np.load('data/train_labels.npy'))
    test_size = len(np.load('data/test_labels.npy'))
    
    start = time.perf_counter()

    # Shared TM class
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError("`args.dtype` must be either `float32` or `float16`")
    num_classes = 10
    tm = ConvTM(classes=num_classes, ks=args.ks, channels=args.channels, num_clauses=250, S=5, T=15, dtype=dtype, device='cuda')

    data_info = {'train_root': 'data/train',
                 'test_root': 'data/test',
                 'train_size': train_size,
                 'test_size': test_size}

    # Hogwild
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=process_main, args=(tm, data_info, rank, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    args.num_processes = 1
    rank = 0
    test_loader = get_final_test_loader(data_info, rank, args)
    test_loop(tm, test_loader, 10)

    print(f'Total time taken: {time.perf_counter()-start} seconds')


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dtype", type=str, default='float32')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
