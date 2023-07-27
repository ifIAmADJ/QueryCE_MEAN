import logging
import os
import time


from torch.utils.data import DataLoader

from lab.models import *
import load_training_set
import numpy as np

def to_cuda(tensors):
    return [t.cuda() for t in tensors]


if __name__ == '__main__':

    from config import *

    level = logging.DEBUG
    logging.basicConfig(level=level)

    # [1/] previous setting
    using = "MEAN"
    training_set_file_path = "../data/imdb_training_100k.csv"
    batch_size = 1024
    lr = 5e-4 # 5e-4 is ok.
    epoch_num = 100
    data_ctx_name = "imdb_bin128"
    take = 0.2
    use_cuda = True

    # [2/] load data, registered data_ctx, and convert to torch.utils.data.DataSet
    data_ctx = register[data_ctx_name]()
    [range_len, proj_shape, tables_len, joins_len] = \
        [data_ctx.ranges_feat_len, data_ctx.proj_feat_shape, data_ctx.tables_feat_len, data_ctx.joins_feat_len]

    logging.info(f"using training data load from : {training_set_file_path}, take {100 * take}% of data.")
    training_dataset = load_training_set.from_csv(
        csv_path=training_set_file_path,
        data_context=data_ctx,
        take=take
    )
    dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size)

    # [3/] choose model
    model = MEAN(proj_feat_shape=proj_shape, tables_num=tables_len, join_pairs_num=joins_len)

    assert isinstance(model, torch.nn.Module), "not a valid model type."
    print(f"using model: {model}")
    model.train()

    if torch.cuda.is_available() and use_cuda:
        print("cuda mode [ON]")
        model.cuda()

    # [4/] choose optimizer and start training
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], lr=lr)
    # set milestones if you need.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[180], gamma=0.5)
    criterion = nn.MSELoss()

    t1 = time.time()
    for epoch in range(epoch_num):
        print(f"epoch : {epoch} training ...", end="")
        epoch_loss = 0.
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # batch = ((feat1, feat2, ...., feat_n), card)
            feats = batch[0]
            cards = batch[1]

            if torch.cuda.is_available() and use_cuda:
                feats = to_cuda(feats)
                cards = cards.cuda()

            outputs = model(*feats)
            loss = criterion(outputs.squeeze(1), cards)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"done, loss = {np.around(epoch_loss / len(dataloader), 3)}")
        scheduler.step()
    t2 = time.time()

    # [5/] save model, exit.
    os.makedirs(f"cached/{data_ctx_name}/{using}", exist_ok=True)
    cache_path = f"cached/{data_ctx_name}/{using}/{batch_size}bs_{lr}lr_ep{epoch_num}_take{take}"
    torch.save(model, cache_path)

    print(f"finish, take {np.round(t2 - t1, decimals=2)}s.")
