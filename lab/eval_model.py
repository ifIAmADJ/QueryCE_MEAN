import functools
import os
import pickle
import time
from typing import List

import numpy as np
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
    model_name = "SCAN"
    testing_csv_name = ["synthetic", "JOB-light", "scale"]
    batch_size = 1024
    epoch_num = 100
    lr = 5e-4
    take = 0.2
    eval_inference_latency = False
    cuda = True

    # [2/7] load trained model
    model_path = f"cached/imdb_bin{bins}/{model_name}/{batch_size}bs_{lr}lr_ep{epoch_num}_take{take}"
    model = torch.load(model_path)
    model.eval()
    model.cpu()
    # if cuda:
    #     model.cuda()

    print(f"using model : {model_path}")
    print(f"model description : {model}")
    print(f"---------------------------\n")


    def eval(testing_csv_name):
        cache_path = f"cached/testing_set/{testing_csv_name}_bin{bins}.pkl"
        # [3/7] encode sql to feature vector
        print(f"testing: {testing_csv_name}")
        dump_time = 0
        start_time = time.time()

        if not eval_inference_latency and os.path.exists(cache_path):
            print("quickly load encoded sql from disk, skip evaluating inference time.")
            with open(cache_path, mode="rb") as f:
                df = pickle.load(f)
        else:
            encoder = sql2tensor(f"imdb_bin{bins}")
            df = pd.read_csv(f"../data/{testing_csv_name}.csv")
            df['sql_encoded'] = df['sql'].map(lambda sql: encoder(sql))
            if not os.path.exists(cache_path):
                with open(cache_path, mode="wb+") as f:
                    pickle.dump(df, f)
                start_time = time.time()

        # [4/7] start inference
        # .map(lambda wrapped_feat: model(*[x.cuda() if cuda else x for x in wrapped_feat])) \
        df['est'] = df['sql_encoded'] \
            .map(lambda wrapped_feat: model(*wrapped_feat)) \
            .map(lambda r: torch.floor(torch.exp(r))) \
            .map(lambda tensor: tensor.squeeze(0).item())
        end_time = time.time()

        # [5/7] eval inference latency if need:
        if eval_inference_latency:
            interval = float(end_time - start_time) * 1000.0
            avg_latency = np.around(interval / len(df), 2)  # test size
            print(f"inference latency: about {avg_latency}ms /query")

        def eval_qerror(output, target):
            return max(output, target) / max(min(output, target), 1)

        df['q_errors'] = df[['label', 'est']].apply(lambda t: eval_qerror(t[0], t[1]), axis=1)
        q_errors = df['q_errors'].map(lambda qe: np.around(qe, 2)).tolist()

        # [6/7] output bad predicts to path: ./lab/results.
        output_results(df)

        # [7/7] output evaluation result to console.
        round2 = functools.partial(np.round, decimals=2)
        print(f"min q-error: {round2(a=np.min(q_errors))}")
        print(f"Median {round2(a=np.median(q_errors))}")
        print(f"90th q-error: {round2(a=np.percentile(q_errors, 90))}")
        print(f"95th q-error: {round2(a=np.percentile(q_errors, 95))}")
        print(f"99th q-error: {round2(a=np.percentile(q_errors, 99))}")
        print(f"max q-error: {round2(a=np.max(q_errors))}")
        print(f"Mean {round2(a=np.mean(q_errors))}")
        print("----------------------------------------------")


    eval(testing_csv_name[0])
    eval(testing_csv_name[1])
    eval(testing_csv_name[2])

