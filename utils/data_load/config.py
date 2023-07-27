COLS = "cols"
UNPICK = "unpick"

# root path of IMDB datasets.
DB_ROOT_PATH = ""

# About IMDB datasets, we just consider about numerical or categorical attributes,
# as same as DeepDB does. =)
tables_and_unpick_cols = {
    "title": {
        COLS: ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
               'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
               'series_years', 'md5sum'],
        UNPICK: ['episode_of_id', 'title', 'imdb_index', 'phonetic_code', 'season_nr',
                 'imdb_id', 'episode_nr', 'series_years', 'md5sum']
    },
    "movie_info_idx": {
        COLS: ['id', 'movie_id', 'info_type_id', 'info', 'note'],
        UNPICK: ['info', 'note']
    },
    "movie_info": {
        COLS: ['id', 'movie_id', 'info_type_id', 'info', 'note'],
        UNPICK: ['info', 'note']
    },
    "cast_info": {
        COLS: ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
               'role_id'],
        UNPICK: ['nr_order', 'note', 'person_role_id']
    },
    "movie_keyword": {
        COLS: ['id', 'movie_id', 'keyword_id'],
        UNPICK: []
    },
    "movie_companies": {
        COLS: ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
        UNPICK: ['note']
    }
}

# ((c1, t1), (c2, t2))  => t1.c1 = t2.c2
join_pairs = [
    (('movie_id', 'movie_info_idx'), ('id', 'title')),
    (('movie_id', 'movie_info',), ('id', 'title')),
    (('movie_id', 'cast_info'), ('id', 'title')),
    (('movie_id', 'movie_keyword'), ('id', 'title')),
    (('movie_id', 'movie_companies'), ('id', 'title'))
]

join_keys = [('movie_id', 'movie_info_idx'), ('movie_id', 'movie_info'), ('movie_id', 'cast_info'),
             ('movie_id', 'movie_keyword'), ('movie_id', 'movie_companies', ('id', 'title'))
]
