# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import scipy
import datetime
from pathlib import Path
from pandas import DataFrame
from fxincome import const, logger


def process_data():

    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    
    # Load bond information to databse
    bond_info = pd.read_csv(const.INDEX_ENHANCEMENT.CDB_INFO, encoding='gbk')
    bond_info.to_sql(
        const.DB.TABLES.IndexEnhancement.CDB_INFO,
        conn,
        if_exists="replace",
        index=False,
    )

    # Load yield spread to database
    yield_spread = pd.read_csv(const.INDEX_ENHANCEMENT.CDB_YC)
    yield_spread.to_sql(
        const.DB.TABLES.IndexEnhancement.CDB_YC,
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()
    return bond_info


if __name__ == "__main__":
    process_data()
