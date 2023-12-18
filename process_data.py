import gc
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
import models

pd.set_option("display.max_columns", None)


def limpia_columnas_hogares(test_hogares_df, train_hogares_df):
    # Identifico las columnas en común entre ambos DataFrames de hogares
    columnas_comunes_hogares = test_hogares_df.columns.intersection(
        train_hogares_df.columns
    )

    # Selecciono solo las columnas comunes en ambos DataFrames
    test_hogares_df = test_hogares_df[columnas_comunes_hogares]
    train_hogares_df = train_hogares_df[
        columnas_comunes_hogares.to_list() + ["Pobre", "Ingpcug"]
    ]

    return test_hogares_df, train_hogares_df


def limpia_columnas_personas(test_personas_df, train_personas_df):
    # Identifico las columnas en común entre ambos DataFrames de personas
    columnas_comunes_personas = test_personas_df.columns.intersection(
        train_personas_df.columns
    )

    # Selecciono solo las columnas comunes en ambos DataFrames
    test_personas_df = test_personas_df[columnas_comunes_personas]
    train_personas_df = train_personas_df[columnas_comunes_personas]

    # Dropeo columnas que están en base hogar (excepto id)
    test_personas_df = test_personas_df.drop(
        columns=["Clase", "Dominio", "Fex_c", "Depto", "Fex_dpto"]
    )
    train_personas_df = train_personas_df.drop(
        columns=["Clase", "Dominio", "Fex_c", "Depto", "Fex_dpto"]
    )

    return test_personas_df, train_personas_df


# Definir una función para calcular el ratio de hijos entre 10 y 18 años que estudian en cada hogar
def calcular_ratio_hijos_estudiando(row):
    # Filtrar las edades entre 10 y 18 años y contar cuántos estudian
    hijos_estudiando = sum(
        (row[f"P6040_{i}"] >= 10) & (row[f"P6040_{i}"] < 18) and row[f"P6240_{i}"] == 3
        for i in range(1, 9)
    )

    # Calcular el ratio
    total_hijos = sum(
        (row[f"P6040_{i}"] >= 10) & (row[f"P6040_{i}"] < 18) for i in range(1, 9)
    )

    # Si no hay hijos en edad de estudiar, devolver NaN
    if total_hijos == 0:
        return np.nan

    return hijos_estudiando / total_hijos


def variables_nuevas(df):
    # Creo una lista con los nombres de las columnas de edad
    columnas_edad = [f"P6040_{i}" for i in range(1, 9)]

    # Filtro el DataFrame para obtener solo las columnas de edad
    df_edades = df[columnas_edad]

    # Creo variables para cantidad de niños y niños en edad escolar
    df["cantidad_ninos"] = df_edades[df_edades < 18].count(axis=1)

    # Prop de niños en el hogar
    df["prop_ninos"] = df["cantidad_ninos"] / df["Nper"]

    # Creo una columna que indique si el jefe de hogar es menor de 25 años y hay al menos un niño en el hogar
    df["padre_joven"] = ((df["P6040_1"] < 25) & (df["cantidad_ninos"] > 0)).astype(int)

    # Creo una columna que indique si hay ancianos viviendo en el hogar (condicional a que el jefe de hogar no lo sea)

    df["ancianos"] = 0
    condicion = df["P6040_1"] < 65
    df.loc[condicion, "ancianos"] = (df_edades[condicion] > 65).any(axis=1).astype(int)

    # Hacinamiento

    df["hacinamiento"] = df["Nper"] / df["P5010"]

    return df


# Definir una función para calcular el ratio de personas que buscan trabajo en cada hogar
def calcular_ratio_buscando_trabajo(row):
    # Contar cuántas personas están buscando trabajo
    personas_buscando_trabajo = sum(row[f"P6240_{i}"] == 2 for i in range(1, 9))

    # Calcular el ratio
    if row["Nper"] == 0:
        return 0  # Evitar división por cero
    return personas_buscando_trabajo / row["Nper"]


# Definir una función para calcular el valor del alquiler por persona
def calcular_alquiler_por_persona(row, columna_pago):
    pago_individual = row[columna_pago]
    n_personas = row["Nper"]

    # Calcular el valor del alquiler por persona
    if pd.isna(pago_individual) or pago_individual == 0 or n_personas == 0:
        return np.nan
    else:
        return pago_individual / n_personas


def agrega_variables(df, type):
    # Aplicar la función a cada fila y agregar una nueva columna 'RatioEstudiando'
    df["RatioEstudiando"] = df.apply(calcular_ratio_hijos_estudiando, axis=1)

    # Aplicar la función a cada fila y agregar una nueva columna 'RatioBuscandoTrabajo'
    df["RatioBuscandoTrabajo"] = df.apply(calcular_ratio_buscando_trabajo, axis=1)

    # Crear nuevas columnas 'AlquilerPorPersona' y 'AlquilerEstimadoPorPersona'
    df["AlquilerPorPersona"] = df.apply(
        calcular_alquiler_por_persona, axis=1, columna_pago="P5100"
    )
    df["AlquilerEstimadoPorPersona"] = df.apply(
        calcular_alquiler_por_persona, axis=1, columna_pago="P5130"
    )

    df = variables_nuevas(df)

    new_cols = [
        "RatioEstudiando",
        "RatioBuscandoTrabajo",
        "AlquilerPorPersona",
        "AlquilerEstimadoPorPersona",
        # De variables_nuevas
        "cantidad_ninos",
        "prop_ninos",
        "padre_joven",
        "ancianos",
        "hacinamiento",
    ]
    if type == "train":
        relevant_vars = ["id", "Pobre", "Ingpcug", "Lp"]
    elif type == "test":
        relevant_vars = ["id", "Lp"]
    df = df[
        [col for col in df.columns if col.endswith("_1")] + new_cols + relevant_vars
    ]
    return df


def standardize(df):
    """
    Standardizes numeric columns in the DataFrame by subtracting the mean and dividing by the standard deviation.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing numeric columns to be standardized.

    Returns:
    - pd.DataFrame: DataFrame with standardized numeric columns.
    """
    numeric = df.select_dtypes(include=["int64", "float64"])
    numeric = numeric.drop(columns=["Lp"])
    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return df


def encode_cat(df):
    """
    Encodes categorical columns in the DataFrame as categorical data types.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing columns to be encoded.

    Returns:
    - pd.DataFrame: DataFrame with categorical columns encoded as categories.
    """
    for col in df.columns:
        if df[col].dtype not in ["int64", "float64", "bool"]:
            df[col] = df[col].astype("category")
        elif df[col].unique().shape[0] < 10:
            df[col] = df[col].astype("category")

    return df


def add_nan_dummies(df):
    """
    Add Boolean columns to the DataFrame indicating whether each value is NaN or not.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with additional Boolean columns indicating NaN values.
    """
    nan_indicators = df.isnull().astype(bool).add_suffix("_is_nan")
    return pd.concat([df, nan_indicators], axis=1)


def pre_process_data(df, type, cols=None, nan=None, cla=True):
    """
    Pre-processes the input DataFrame by standardizing numeric columns, encoding categorical variables,
    and performing additional data cleaning steps.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to be pre-processed.
    - type (str, "train" or "test"): Wheter the dataframe passed is test or train
    - impute (bool, optional): Whether to perform imputation for missing values. Default is False.

    Returns:
    - pd.DataFrame: Pre-processed DataFrame with standardized and encoded features.
    """
    print("Input shape:\t{}".format(df.shape))

    # Encode categorical vars:
    df = df.set_index("id")
    df = encode_cat(df)
    if type == "train":
        if cla is True:
            y = df["Pobre"].astype(bool)
        else:
            y = df[["Ingpcug", "Lp"]].astype("float32")
        df = pd.get_dummies(df.drop(columns=["Pobre", "Ingpcug"]), drop_first=True)
    else:
        df = pd.get_dummies(df, drop_first=True)

    df = df.dropna(axis=1, how="all")

    # Standarize vars:
    df = standardize(df)

    # Imputation ¿we want this?
    if nan == "impute":
        df = df.fillna(df.median())
    elif nan == "dummies":
        df = add_nan_dummies(df)
        df = df.fillna(0)
    else:
        df = df.fillna(0)
    print("Final shape {}".format(df.shape))

    # Dropea columnas no existentes en train y genera las faltantes
    if cols is not None:
        common_cols = df.columns.intersection(cols)
        df = df[common_cols]
        missing_cols = cols.difference(df.columns)
        df[missing_cols] = np.nan
        df[
            [
                col
                for col in df.columns
                if col.endswith("_is_nan") and col in missing_cols
            ]
        ] = True
        df[missing_cols] = 0

    # To float - for keras compatibility
    X = df.astype("float32").reset_index(drop=True)
    if type == "train":
        y = y.astype("float32").reset_index(drop=True)
    else:
        y = None
    return X, y


def build_dataset_A(
    train_personas_df,
    train_hogares_df,
    test_personas_df,
    test_hogares_df,
    nan="dummies",
    cla=True,
):
    """Solo Jefes de hogar"""

    def build_dataset_A_pipe(personas, hogares, type, nan, cla=True, cols=None):
        jefes = personas[personas["Orden"] == 1]
        df = hogares.merge(jefes, on="id")
        df = df.drop(columns=[col for col in df.columns if col.endswith("_y")])
        df = df.rename(
            columns={col: col[:-2] for col in df.columns if col.endswith("_x")}
        )
        X, y = pre_process_data(df, cols=cols, type=type, nan=nan, cla=cla)
        return X, y

    X, y = build_dataset_A_pipe(
        train_personas_df,
        train_hogares_df,
        type="train",
        nan=nan,
        cla=cla,
    )
    X_test, _ = build_dataset_A_pipe(
        test_personas_df, test_hogares_df, cols=X.columns, type="test", nan=nan
    )

    return X, y, X_test


def build_dataset_B(
    train_personas_df,
    train_hogares_df,
    test_personas_df,
    test_hogares_df,
    nan="dummies",
    cla=True,
):
    """Covariables de todos los convivientes"""

    def build_dataset_B_pipe(personas, hogares, type, nan, cols=None, cla=cla):
        parejas = personas[personas["Orden"] <= 8]

        # Pivot the DataFrame using pivot_table
        parejas = parejas.set_index(["id", "Orden"])
        parejas = parejas.unstack(level=-1)

        # Rename columns
        parejas.columns = [
            "{}_{}".format(level0, level1) if level1 != "" else level0
            for level0, level1 in parejas.columns
        ]
        df = hogares.merge(parejas, on="id")
        X, y = pre_process_data(df, cols=cols, type=type, nan=nan, cla=cla)

        return X, y

    X, y = build_dataset_B_pipe(
        train_personas_df, train_hogares_df, type="train", nan=nan, cla=cla
    )
    X_test, _ = build_dataset_B_pipe(
        test_personas_df, test_hogares_df, type="test", nan=nan
    )

    return X, y, X_test


def build_dataset_C(
    train_personas_df,
    train_hogares_df,
    test_personas_df,
    test_hogares_df,
    nan="dummies",
    cla=True,
):
    """Jefes + data agregada por hogar"""

    def build_dataset_C_pipe(personas, hogares, type, nan, cols=None, cla=cla):
        # Pivot the DataFrame using pivot_table
        personas = personas.set_index(["id", "Orden"])
        personas = personas.unstack(level=-1)

        # Rename columns
        personas.columns = [
            "{}_{}".format(level0, level1) if level1 != "" else level0
            for level0, level1 in personas.columns
        ]
        df = hogares.merge(personas, on="id")
        df = df.drop(columns=[col for col in df.columns if col.endswith("_y")])
        df = df.rename(
            columns={col: col[:-2] for col in df.columns if col.endswith("_x")}
        )

        # Agrega variables gruopales y limpia data de personas
        df = agrega_variables(df, type=type)

        X, y = pre_process_data(df, cols=cols, type=type, nan=nan, cla=cla)
        return X, y

    X, y = build_dataset_C_pipe(
        train_personas_df, train_hogares_df, type="train", nan=nan, cla=cla
    )
    X_test, _ = build_dataset_C_pipe(
        test_personas_df, test_hogares_df, cols=X.columns, type="test", nan=nan
    )

    return X, y, X_test
