import utils.data_load as dl


# for lazy load.
register = {
    "imdb_bin128":
        lambda: dl.apply_db_context(db_name="imdb", bins=128, dumping_path="../schema_meta"),
}
