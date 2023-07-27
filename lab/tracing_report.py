import functools
import os
import pickle
import time
from typing import List

import numpy as np
import pandas
import pandas as pd
import torch

import utils.data_load as dl
from lab.config import register


def sql2tensor(registered_table_or_db_name: str):
    transformer: dl.AbstractSqlEncoder = register[registered_table_or_db_name]()

    def f(sql: str) -> List[torch.Tensor]:
        feat_sequence = transformer.encode_sql(sql)
        return [torch.tensor(feat, dtype=torch.float32).unsqueeze(0) for feat in feat_sequence]

    return f


def output_results(q_errors_df: pd.DataFrame):
    poor_predict_tau = 50
    bad_predict_tau = 100
    terrible_predict_tau = 500
    good_predicts = q_errors_df[q_errors_df['q_errors'] < 5][['sql', 'label', 'est', 'q_errors']].head(100)
    poor_predicts = \
    q_errors_df[(q_errors_df['q_errors'] > poor_predict_tau) & (q_errors_df['q_errors'] < bad_predict_tau)][
        ['sql', 'label', 'est', 'q_errors']]
    small_scale_predicts = q_errors_df[(q_errors_df['label'] < 30)][['sql', 'label', 'est', 'q_errors']]
    bad_predicts = \
        q_errors_df[(bad_predict_tau < q_errors_df['q_errors']) & (terrible_predict_tau > q_errors_df['q_errors'])] \
            [['sql', 'label', 'est', 'q_errors']]
    terrible_predicts = q_errors_df[q_errors_df['q_errors'] > terrible_predict_tau][['sql', 'label', 'est', 'q_errors']]

    if len(terrible_predicts) != 0:
        print(
            f"detect {len(bad_predicts)} bad predicts that q-error ranges in [{bad_predict_tau}, {terrible_predict_tau}]")
        print(f"detect {len(terrible_predicts)} terrible predicts that q-error > {terrible_predict_tau}")

    poor_predicts.to_csv(f"results/poor_predicts.csv", index=False, mode="w")
    small_scale_predicts.to_csv(f"results/small_scale_predicts.csv", index=False, mode="w")
    good_predicts.to_csv(f"results/good_predicts.csv", index=False, mode="w")
    bad_predicts.to_csv(f"results/bad_predicts.csv", index=False, mode="w")
    terrible_predicts.to_csv(f"results/terrible_predicts.csv", index=False, mode="w")


if __name__ == '__main__':
    # [1/7] The encoder will load the data set already registered in dict 'registered',
    # and then automatically translate the sql(s) to the input(s) of model.
    bins = 128
    model_name = "MEAN"
    testing_csv_name = ["synthetic", "JOB-light", "scale"]
    batch_size = 1024
    epoch_num = 100
    lr = 5e-4
    take = 0.2
    eval_inference_latency = True

    # [2/7] load trained model
    model_path = f"cached/imdb_bin{bins}/{model_name}/{batch_size}bs_{lr}lr_ep{epoch_num}_take{take}"
    model = torch.load(model_path)
    model.eval()
    model.cpu()

    print(f"using model : {model_path}")
    print(f"model description : {model}")
    print(f"---------------------------\n")

    encoder = sql2tensor(f"imdb_bin{bins}")
    for workload in testing_csv_name:
        df = pandas.read_csv(f"../data/{workload}.csv")
        seq = df.to_numpy().tolist()
        size = len(seq)

        total_latency_time = 0

        def q_error(y1, y2):
            return max(y1, y2) / min(y1, y2)

        from collections import namedtuple
        Result = namedtuple('Result',["sql","join_num", "errs", "est", "real", "isUnderEstimated"])
        res_seq : List[Result] = []

        from tqdm import tqdm

        pbar = tqdm(total=size)
        for sql, label in seq:
            # print(f"sql: {sql}, and label: {label}", end=" ")
            feats = encoder(sql)
            # print("feats =", *feats)

            t1 = time.time()
            result = torch.exp(model(*feats))

            # the last of feats has record the number of tables, join_num = table_nun - 1.
            join_num = ((feats[-1] != 0).sum(dim=1) - 1).detach().numpy()[0]
            latency = time.time() - t1
            # Remove redundant dimensions, e.g. tensor([[10]]) -> 10
            result = result.squeeze(dim=1).detach().numpy()[0]
            q_err = q_error(result, label)

            total_latency_time += latency
            # print(f"result = {result}, join_num = {join_num}, q-error: {q_err}, latency = {latency * 1000}ms")
            res_seq += [Result(join_num=join_num, errs=q_err, est= result, real=label, isUnderEstimated= result < label, sql =sql)]
            pbar.update(1)
        print(f"avg latency = {total_latency_time / len(seq) * 1000}ms")

        df = pandas.DataFrame(res_seq, columns=["sql","join_num", "errs", "est", "real", "isUnderEstimated"])
        # df.to_csv("results/report_MEAN.csv", index=False)

        q_errors = df['errs']
        per99 = np.percentile(q_errors, 99)
        remain = q_errors[q_errors < per99]
        res = np.mean(remain)
        max1 = np.max(q_errors)
        print(f"epoch = {epoch_num}, training_size = {take * 100_000}, workload = {workload}")
        print(f"99% Mean Q-Error = {res}")
        print(f"Max Q-Error = {max1}")
        print(f"--------------------------")
    print("done.")