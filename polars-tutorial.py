# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # My Small [Polars](https://www.pola.rs/) Tutorial
#
# [This](https://github.com/sotte/polars-tutorial) is a small polars tutorial.
# It covers basic concepts as well as some random (hopefully) useful things.
#
# It is based on the great
# [polars cheat sheet](https://franzdiebold.github.io/polars-cheat-sheet/Polars_cheat_sheet.pdf)
# but is heavily extended and restructured.

# %% [markdown]
# Here are some important facts that you should know when you work with polars.
#
# **polars DataFrames don't have an index**. Aka no `.reset_index()` every third line and less complexity due to no multi-index,
# but some operartions are a bit more cumbersome in return.
#
# There are two main concepts in polars: expressions and contect.
#
# - **expression**: what to do with the data without actually knowing the data, e.g. `pl.col("foo").sort().head(2)`
# - **context**: the context in which an expression is evaluated, e.g. in a `group_by`
#
# Because of the expression/context setup **method chaining** makes even more sense than with pandas.
#
# These seven verbs cover most things you want to do with polars:
#
# ```python
# select        # select columns (and add new ones)
# with_columns  # like select but keep existing columns
# sort          # sort rows
# filter        # filter rows
# group_by      # group dataframe
# agg           # aggregate groups
# join          # join/merge another dataframe
# ```
#
# As always, [read the friendly manual](https://docs.pola.rs/) to really understand how to use polars.
#
# IMO reading manuals is a super power and everybody can attain it. :shrug:

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Install

# %%
# !pip install polars==0.20.25

# we'll install a bit more for later parts of the tutorial

# we want to plot dataframes
# !pip install "polars[plot,pyarrow]==0.20.25"

# we want to validate dataframes with pandera
# !pip install "pandera[polars]==0.19.2"

# %% [markdown]
# ### Import

# %% pycharm={"is_executing": true}
import polars as pl

# I personally like to import col as c to shorten some expressions
from polars import col as c

# %% [markdown]
# ## Basics

# %% [markdown]
# ### Creating/Reading/Saving DataFrames

# %%
# Create DataFrame
df = pl.DataFrame(
    {
        "nrs": [1, 2, 3, None, 5],
        "names": ["foo", "ham", "spam", "egg", None],
        "random": [0.3, 0.7, 0.1, 0.9, 0.6],
        "groups": ["A", "A", "B", "C", "B"],
    }
)
df

# %%
# Save dataframes as csv
# (do yourself a favour and switch to parquet instead of CSV!)
df.write_csv("df.csv")

# %%
# Read CSV
(pl.read_csv("df.csv", columns=["nrs", "names", "random", "groups"]).equals(df))

# %%
# Save dataframe as parquet
df.write_parquet("df.parquet")

# %%
# Read parquet
# Note: you can also read multiple frames with wildcards
(pl.read_parquet("df*.parquet").equals(df))

# %% [markdown]
# ### Select columns - `select()`

# %%
# Select multiple columns with specific names
df.select("nrs", "names")
# equivalent
df.select(pl.col("nrs"), pl.col("names"))
df.select(pl.col.nrs, pl.col.names)

# %%
df.select(pl.all().exclude("random"))

# %%
# Select columns whose name matches regular expression regex.
df.select(pl.col("^n.*$"))

# %% [markdown]
# ### Add New Columns - `select()` and `with_columns()`

# %%
df.select(NAMES=c.names)
df.select(c("names").alias("NAMES"))

# %%
# Keep existing and add new columns with `with_columns`
df.with_columns((pl.col("random") * pl.col("nrs")).alias("product"))
df.with_columns(product=pl.col("random") * pl.col("nrs"))

# %%
# Add several new columns to the DataFrame
df.with_columns(
    product=(pl.col("random") * pl.col("nrs")),
    names_len_bytes=pl.col("names").str.len_bytes(),
)

# %%
# Add a column 'index' that enumerates the rows
df.with_row_index()

# %% [markdown]
# ### Select rows - `filter()` and friends

# %%
# Filter: Extract rows that meet logical criteria.
df.filter(pl.col("random") > 0.5)
df.filter(c("random") > 0.5)
df.filter(c.random > 0.5)

# %%
df.filter((pl.col("groups") == "B") & (pl.col("random") > 0.5))

# %%
# Randomly select fraction of rows.
df.sample(fraction=0.5)

# %%
# Randomly select n rows.
df.sample(n=2)

# %%
# Select first n rows
df.head(n=2)

# Select last n rows.
df.tail(n=2)

# %% [markdown]
# ### Select rows and columns

# %%
# Select rows 2-4
df[2:4, :]

# %%
# Select columns in positions 1 and 3 (first column is 0).
df[:, [1, 3]]

# %%
# Select rows meeting logical condition, and only the specific columns.
(df.filter(pl.col("random") > 0.5).select("names", "groups"))

# %%
# Select one columns as Series
print(type(df["names"]))
df["names"]

# %% [markdown]
# ### Sort rows - `sort()`

# %%
# Order rows by values of a column (high to low)
df.sort("random", descending=True)

# %%
# Order by multiple rows
df.sort("groups", "random")

# %% [markdown]
# ### Summarize Data

# %%
# Tuple of # of rows, # of columns in DataFrame
df.shape

# %%
# number of rows in DataFrame
len(df)
df.height

# %%
# number of cols in DataFrame
df.width

# %%
# Count number of rows with each unique value of variable
df["groups"].value_counts()

# %%
# # of distinct values in a column
df["groups"].n_unique()

# %%
# Basic descriptive and statistics for each column
df.describe()

# %%
# Aggregation functions
df.select(
    # Sum values
    pl.sum("random").alias("sum"),
    # Minimum value
    pl.min("random").alias("min"),
    # Maximum value
    pl.max("random").alias("max"),
    # or
    pl.col("random").max().alias("other_max"),
    # Standard deviation
    pl.std("random").alias("std_dev"),
    # Variance
    pl.var("random").alias("variance"),
    # Median
    pl.median("random").alias("median"),
    # Mean
    pl.mean("random").alias("mean"),
    # Quantile
    pl.quantile("random", 0.75).alias("quantile_0.75"),
    # or
    pl.col("random").quantile(0.75).alias("other_quantile_0.75"),
    # First value
    pl.first("random").alias("first"),
)

# %% [markdown]
# ### Group And Aggregate Data - `group_by()` and `agg()`

# %%
# Group by values in column named "col", returning a GroupBy object
df.group_by("groups")

# %%
# All of the aggregation functions from above can be applied to a group as well
df.group_by("groups").agg(
    # Sum values
    pl.sum("random").alias("sum"),
    # Minimum value
    pl.min("random").alias("min"),
    # Maximum value
    pl.max("random").alias("max"),
    # or
    pl.col("random").max().alias("other_max"),
    # Standard deviation
    pl.std("random").alias("std_dev"),
    # Variance
    pl.var("random").alias("variance"),
    # Median
    pl.median("random").alias("median"),
    # Mean
    pl.mean("random").alias("mean"),
    # Quantile
    pl.quantile("random", 0.75).alias("quantile_0.75"),
    # or
    pl.col("random").quantile(0.75).alias("other_quantile_0.75"),
    # First value
    pl.first("random").alias("first"),
)

# %%
# Additional GroupBy functions
(
    df.group_by("groups").agg(
        # Count the number of values in each group
        pl.count("random").alias("size"),
        # Sample one element in each group
        # (favour `map_elements` over `apply`)
        pl.col("names").map_elements(
            lambda group_df: group_df.sample(1).item(0), return_dtype=pl.String
        ),
    )
)

# %%
(df.group_by("groups").agg(pl.col("names").sample(1).alias("foo")))

# %% [markdown]
# ### Reshaping Data – Change Layout and Renaming

# %%
# Rename the columns of a DataFrame
df.rename({"nrs": "idx"})

# %%
# Drop columns from DataFrame
df.drop(["names", "random"])

# %%
df2 = pl.DataFrame(
    {
        "nrs": [6],
        "names": ["wow"],
        "random": [0.9],
        "groups": ["B"],
    }
)

df3 = pl.DataFrame(
    {
        "primes": [2, 3, 5, 7, 11],
    }
)

# %%
# Append rows of DataFrames.
pl.concat([df, df2])

# %%
# Append columns of DataFrames
pl.concat([df, df3], how="horizontal")

# %%
# Gather columns into rows
df.melt(id_vars="nrs", value_vars=["names", "groups"])

# %%
# Spread rows into columns
df.pivot(values="nrs", index="groups", columns="names")

# %% [markdown]
# ### Reshaping Data - Join Data Sets

# %%
df4 = pl.DataFrame(
    {
        "nrs": [1, 2, 5, 6],
        "animals": ["cheetah", "lion", "leopard", "tiger"],
    }
)

# %%
# Inner join
# Retains only rows with a match in the other set.
df.join(df4, on="nrs")
# or
df.join(df4, on="nrs", how="inner")

# %%
# Left join
# Retains each row from "left" set (df).
df.join(df4, on="nrs", how="left")

# %%
# Outer join
# Retains each row, even if no other matching row exists.
df.join(df4, on="nrs", how="outer")

# %%
# Anti join
# Contains all rows from df that do not have a match in df4.
df.join(df4, on="nrs", how="anti")

# %% [markdown]
# ## Misc

# %% [markdown]
# ### Handling Missing Data

# %%
# How many nulls per column?
df.null_count()

# %%
# Drop rows with any column having a null value
df.drop_nulls()

# %%
# Replace null values with given value
df.fill_null(42)

# %%
# Replace null values using forward strategy
df.fill_null(strategy="forward")
# Other fill strategies are "backward", "min", "max", "mean", "zero" and "one"

# %%
# Replace floating point NaN values with given value
df.fill_nan(42)

# %% [markdown]
# ### Rolling Functions

# %%
# The following rolling functions are available
import numpy as np

df.select(
    pl.col("random"),
    # Rolling maximum value
    pl.col("random").rolling_max(window_size=2).alias("rolling_max"),
    # Rolling mean value
    pl.col("random").rolling_mean(window_size=2).alias("rolling_mean"),
    # Rolling median value
    pl.col("random")
    .rolling_median(window_size=2, min_periods=2)
    .alias("rolling_median"),
    # Rolling minimum value
    pl.col("random").rolling_min(window_size=2).alias("rolling_min"),
    # Rolling standard deviation
    pl.col("random").rolling_std(window_size=2).alias("rolling_std"),
    # Rolling sum values
    pl.col("random").rolling_sum(window_size=2).alias("rolling_sum"),
    # Rolling variance
    pl.col("random").rolling_var(window_size=2).alias("rolling_var"),
    # Rolling quantile
    pl.col("random")
    .rolling_quantile(quantile=0.75, window_size=2, min_periods=2)
    .alias("rolling_quantile"),
    # Rolling skew
    pl.col("random").rolling_skew(window_size=2).alias("rolling_skew"),
    # Rolling custom function
    pl.col("random")
    .rolling_map(function=np.nanstd, window_size=2)
    .alias("rolling_apply"),
)

# %% [markdown]
# ### Window Functions

# %%
# Window functions allow to group by several columns simultaneously
df.select(
    "names",
    "groups",
    "random",
    pl.col("random").sum().over("names").alias("sum_by_names"),
    pl.col("random").sum().over("groups").alias("sum_by_groups"),
)

# %% [markdown]
# ### Date Range Creation

# %%
# create data with pl.date_range
from datetime import date


df_date = pl.DataFrame(
    {
        # eager=True is important to turn the expression into actual data
        "date": pl.date_range(
            date(2024, 1, 1), date(2024, 1, 7), interval="1d", eager=True
        ),
        "value": [1, 2, 3, 4, 5, 6, 7],
    }
)
df_date

# %% [markdown]
# ### Time-based Upsampling - `group_by_dynamic()`
# `group_by_dynamic` has **many** useful options.
#
# [[docs]](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html#polars.DataFrame.group_by_dynamic)

# %%
(df_date.group_by_dynamic("date", every="1w").agg(c.value.sum()))


# %% [markdown]
# ### Custom Expressions
# You can create your own expression and reuse them throughout your projects.


# %%
def normalize_str_col(col_name: str) -> pl.Expr:
    return pl.col(col_name).str.to_lowercase().str.replace_all(" ", "_")


df.select(new_col=normalize_str_col(col_name="groups"))

# %% [markdown]
# ## Plotting
# Polars does not implement plotting itself, but delegates to `hvplot`.
# The plot functions are available via the `.plot` accessor.
#
# [[docs]](https://docs.pola.rs/py-polars/html/reference/dataframe/plot.html)

# %%
df

# %%
(
    df.group_by("groups", maintain_order=True)
    .agg(pl.sum("random"))
    .plot.bar(x="groups", y="random")
)


# %% [markdown]
# ## Debugging
# If you have a long chain of transformations, it can be handy to look at / log intemediate steps.
# Write little helpers to make this easy and call them via `pipe()`.


# %%
def log_df(df: pl.DataFrame, prefix="") -> pl.DataFrame:
    print(f"{prefix}shape:{df.shape}  schema: {dict(df.schema)}")
    return df


(
    df.pipe(log_df, "step 1: ")
    .filter(c.random > 0.5)
    .pipe(log_df, "step 2: ")
    .select("names")
    .pipe(log_df, "step 3: ")
)

# %% [markdown]
# ## Eager, lazy, out-of-core
# Lazy mode allows optimization of the query plan.
#
# Out-of-core or streaming allows to work with data that is bigger than the RAM.

# %%
# eager loading, lazy execution
data = pl.read_parquet("df.parquet")
(
    data.lazy()
    .group_by("groups")
    .agg(pl.sum("random").alias("total"))
    .sort("total")
    # till here nothing really happened
    .collect()  # now we execute the plan and collect the results
)

# %%
# lazy loading, lazy execution
data = pl.scan_parquet("df.parquet")
(
    data.lazy()
    .group_by("groups")
    .agg(pl.sum("random").alias("total"))
    .sort("total")
    # till here nothing really happened
    # with the next line, we execute the plan and collect the results
    .collect()
)

# %%
# stream data
data = pl.scan_parquet("df.parquet")
(
    data.lazy()
    .group_by("groups")
    .agg(pl.sum("random").alias("total"))
    .sort("total")
    # till here nothing really happened
    # with the next line, we execute the plan in a streaming fashion
    .collect(streaming=True)
)

# %% [markdown]
# ## Data Validation with Pandera
#
# Since 0.19 [pandera offers polars support](https://pandera.readthedocs.io/en/stable/polars.html).
# That means you can validate the schema and data of your polars DataFrame.
#
# This is just a sneak peak, read the docs for more.

# %%
import pandera.polars as pa


# define your schema in as much detail as you want


class MySchema(pa.DataFrameModel):
    nrs: int
    names: str  # or pl.String
    # different range
    random: float = pa.Field(in_range={"min_value": 1.0, "max_value": 2.0})
    # C is not allowed
    groups: str = pa.Field(isin=["A", "B"])

    class Config:
        # All existing columns must be listed in the schema
        strict = True


# %%
# Then validate it.
# Use lazy=True to run all validations before throwing the SchemaErrors
try:
    MySchema.validate(df, lazy=True)
except pa.errors.SchemaErrors as e:
    print("Got SchemaErrors exception.")
    print(e)
